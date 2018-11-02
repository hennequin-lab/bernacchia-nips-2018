open Printf
open Owl
open Util
open Dir
module Mat = Dense.Matrix.S

let input_size = 784
let output_size = input_size
let record_every = 20000

module N = Network.Make (struct
    let input_size = input_size
    let layer_sizes = [| 400; 200; 100; 50; 100; 200; 400; input_size |]
    let act_funs = Network.[| tanh; tanh; tanh; tanh; tanh; tanh; tanh; linear |]
  end)

module L = Loss.Squared

let nn0 = N.make_random ()

let (mu, scaling), train_set =
  let x, _, _ = Dataset.load_mnist_train_data () in
  let mu = Mat.mean ~axis:0 x in
  let x = Mat.(x - mu) in
  let scaling = 1. /. Mat.std' x in
  let x = Mat.(scaling $* x) in
  (mu, scaling), (x, x)

let _ = printf "NUMBER = %f\n%!" (0.5 *. Mat.l2norm_sqr' (fst  train_set) /. float (Mat.row_num (fst train_set)))

let train_set_size = Mat.row_num (fst train_set)

let test_set = 
  let x, _, _ = Dataset.load_mnist_test_data () in
  let x = Mat.(scaling $* (x-mu)) in
  x, x

let do_for (module O: Optimizer.T with type net = N.t) n_iter_through_training_set minibatch_size prefix =
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
  let rec iterate test_error cpu t = 
    if t < t_max then begin
      printf "\rt = %7i%!" t;
      Gc.minor ();
      let mb = get_minibatch train_set minibatch_size in
      let _, info = O.step ~t net ~update:true mb in
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
      iterate test_error cpu (t+1)
    end in
  iterate [] [] 0;
  N.error (module L) net test_set


let _ = if Cmdargs.check "-sgd" then begin
    let n_iter_through_training_set = 80 in
    List.iter (fun alpha ->
        let module O = Sgd.Make (struct
            module N = N
            module L = L
            let alpha = alpha
          end) in
        let minibatch_size = 20 in
        let folder = sprintf "sgd_alpha_%.4f/" alpha in
        do_for (module O) n_iter_through_training_set minibatch_size folder |> ignore
      ) [ 0.04; 0.06; 0.08 ]
  end

let _ = if Cmdargs.check "-ng" then
    let n_iter_through_training_set = 20 in
    let minibatch_size = 1000 in
    let module O = Ngd.Make (struct
        module N = N
        module L = L
        include Ngd.Default_prms
        let alpha = 1.0
        let damping = `Levenberg 1.
        let n_repeats = 1
      end) in
    do_for (module O) n_iter_through_training_set minibatch_size "ngd/" |> ignore

