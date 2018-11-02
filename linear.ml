open Owl
open Printf
open Util
open Dir

let n_iter_through_training_set = 40
let train_set_size = 10000
let test_set_size = 1000
let input_size = 20 (* 200 for last network *)
let record_every = 1000
let with_input_cor = true

module N = Network.Make (struct
    let input_size = input_size
    let layer_sizes = [| 20; 20; 20; 20; 20; 20; 20; 20; 20; 20; 20; 20; 20; 20; 20; 20 |] (* NET 1 *)
    (* let layer_sizes = [| 30; 40; 50; 60; 70; 80; 90; 100; 90; 80; 70; 60; 50; 40; 30; 20 |] *) (* NET 2 *)
    (* let layer_sizes = [| 80; 34; 20; 10; 5; 2; 5; 10; 20; 34; 80; 200 |] *) (* NET 3 *)
    let act_funs = Array.map (fun _ -> Network.linear) layer_sizes
  end)

module L = Loss.Squared

let reference = N.make_random ()
let nn0 = N.make_random ~sigmas:(Array.make (N.n_layers reference) 1.) ()
let _ = printf "TOTAL PARAMS = %i | SHALLOW: %i\n%!" (N.n_prms reference) (input_size * input_size) 

let draw_input = 
  if with_input_cor then cor_noise ~epsilon:1e-3 ~sigma:1. ~rank:5. input_size
  else fun size -> Mat.gaussian size input_size

let train_set = 
  let x = draw_input train_set_size in
  let ytilde = N.run reference x in
  let n, m = Mat.shape ytilde in
  let ytilde = Mat.(ytilde + gaussian ~sigma:0.001 n m) in
  x, ytilde

let test_set = 
  let x = draw_input test_set_size in
  let ytilde = N.run reference x in
  let n, m = Mat.shape ytilde in
  let ytilde = Mat.(ytilde + gaussian ~sigma:0.001 n m) in
  x, ytilde

let do_for (module O: Optimizer.T with type net = N.t) minibatch_size prefix =
  let in_dir s = in_dir (sprintf "%s%s" prefix s) in
  if prefix <> "" then begin try Unix.mkdir (in_dir "") 0o777 with _ -> () end;
  let net = N.copy nn0 in
  let prev_net = N.copy net in
  let err0 = N.error (module L) net test_set in
  printf "err0 = %f\n%!" err0;
  let t_max = n_iter_through_training_set * train_set_size / minibatch_size in
  printf "t_max = %i\n%!" t_max;
  let time_to_remove = ref 0. in
  let t_start = Unix.gettimeofday () in
  let rec iterate test_error cpu t info = 
    if t < t_max then begin
      printf "\rt = %7i%!" t;
      Gc.minor ();
      let test_error, cpu = if t mod (record_every / minibatch_size) = 0 then begin
          let result, tz = time_this (fun () ->
              let tee = N.error (module L) net test_set in
              let total_time = Unix.gettimeofday () -. t_start in
              let wallclock = total_time -. !time_to_remove in
              printf "\rt = %7i | test: %.5f | wallclock: %f | %s%!" t tee wallclock info;
              N.copy_into net prev_net;
              let test_error = tee :: test_error in
              let cpu = wallclock :: cpu in
              (* N.save net in_dir; *)
              save test_error  (in_dir "test");
              save cpu (in_dir "cpu_time");
              test_error, cpu) in
          time_to_remove := tz +. !time_to_remove;
          result
        end else (test_error, cpu) in
      let mb = get_minibatch train_set minibatch_size in
      let _, info = O.step ~t net ~update:true mb in
      iterate test_error cpu (t+1) info
    end in
  iterate [] [] 0 "";
  N.error (module L) net test_set


let _ =  (* compute lower-bound on test error using reference network *) 
  let best_test_err = N.error (module L) reference test_set in
  Mat.(save_txt (create 100 1 best_test_err) (in_dir "test_lb"))


(* run SGD *)
let _ = if Cmdargs.check "-sgd" then begin
    let minibatch_size = 20 in
    List.iter (fun alpha ->
        let module O = Sgd.Make (struct
            module N = N
            module L = L
            let alpha = alpha
          end) in
        do_for (module O) minibatch_size (sprintf "sgd_alpha_%.2f/" alpha) |> ignore
      ) [ 0.02; 0.04; 0.06; 0.08 ]
  end

(* run NGD *)
let _ = if Cmdargs.check "-ng" then
    let minibatch_size = 1000 in
    let module O = Ngd.Make (struct
        module N = N
        module L = L
        include Ngd.Default_prms
        let alpha = 1.0
        let damping = `Levenberg 0.01
        let n_repeats = 1
      end) in
    do_for (module O) minibatch_size "ngd/" |> ignore



