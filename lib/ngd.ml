open Printf
open Bigarray
open Owl
open Util

module Default_prms = struct
  let alpha = 1.
  let n_repeats = 10
  let damping = `Levenberg 1E-5
end

module Make (P: sig
    module N: Network.T
    module L: Loss.T
    val alpha : float
    val damping: [ `Fixed of float | `Levenberg of float ]
    val n_repeats: int
  end) : (Optimizer.T with type net = P.N.t) = struct

  open P
  type net = N.t
  let damp = ref (match damping with
      | `Fixed d -> d
      | `Levenberg d -> d)

  type cache = { 
    prev_w: Mat.mat; 
    g_repeats: Mat.mat;
    inp_ng: Mat.mat;
    buf: Mat.mat;
    buf_inp: Mat.mat;
    buf_adj: Mat.mat;
  }

  let get_cache =
    let __cache = Hashtbl.create 1 in
    fun data_size ->
      try Hashtbl.find __cache data_size 
      with Not_found -> 
        let net = N.make_random () in
        let c = Array.mapi (fun id l ->
            let a, b = Mat.shape l.Network.w in
            { 
              prev_w = Mat.empty a b;
              g_repeats = Mat.empty (data_size * n_repeats) b;
              inp_ng = Mat.empty data_size a;
              buf = Mat.zeros a b;
              buf_inp = Mat.zeros data_size a; 
              buf_adj = Mat.zeros data_size b;
            }
          ) (N.layers net) in
        Hashtbl.add __cache data_size c;
        c

  let reset_cache c = ()

  let step ~t net ~update (x, ytilde) =
    let data_size, _ = Mat.shape x in
    let c = get_cache data_size in
    let nc, loss_buf = N.get_cache net data_size in

    (* first, update the estimation of the inverse Fisher bits,
       based on several reverse passes with different noise realisations *)
    N.forward_pass net x; (* do a forward pass once and for all *)
    for k=0 to n_repeats-1 do
      let into = (last nc).da in
      Mat.(copy_ ~out:into (gaussian ~sigma:1. (row_num into) (col_num into)));
      N.backward_pass net data_size;
      Array.iteri (fun i nci ->
          let ci = c.(i) in
          Genarray.blit nci.Network.inp ci.inp_ng;
          Genarray.blit nci.Network.g (rows ci.g_repeats (k*data_size) data_size);
        ) nc;
    done;

    (* now, take a natural gradient step *)
    let y = (last nc).a in
    if update then begin
      (* N.forward_pass net x; (* do a forward pass once and for all *) *)
      let loss = L.loss ~buf:loss_buf ~y ~ytilde in
      L.dloss ~into:(last nc).da ~y ~ytilde;
      N.backward_pass net Mat.(row_num x);
      let predicted_red = ref 0. in
      Array.iteri (fun i li ->
          Gc.minor ();
          let alpha = alpha /. float (N.n_layers net) in
          let nci = nc.(i) in
          let inp = nci.inp in
          let adj = nci.g in
          let sqrt_damp = sqrt !damp in
          let inp_ng = c.(i).inp_ng in
          let _ = 
            let s, vt = truncated_svd ~tol:0. inp_ng in
            Mat.sqr_ s;
            Mat.scalar_mul_ (1. /. float (Mat.row_num inp_ng)) s;
            let s_inv = s |> Mat.(($+) sqrt_damp) |> Mat.reci in
            Genarray.blit Mat.(((inp *@ (transpose vt)) * s_inv) *@ vt) c.(i).buf_inp in
          let _ = 
            let adj_ng = c.(i).g_repeats in
            let s, vt = truncated_svd ~tol:0. adj_ng in
            Mat.sqr_ s;
            Mat.scalar_mul_ (1. /. float (Mat.row_num adj_ng)) s;
            let s_inv = s |> Mat.(($+) sqrt_damp) |> Mat.reci in
            Genarray.blit Mat.(((adj *@ (transpose vt)) * s_inv) *@ vt) c.(i).buf_adj in
          (* compute natural gradient -- cf. Eq 10 *)
          gemm ~transa:true c.(i).buf_inp c.(i).buf_adj c.(i).buf;
          Mat.scalar_mul_ alpha c.(i).buf;
          Genarray.blit li.Network.w c.(i).prev_w; (* save a copy in case things go wrong *)
          Mat.sub_ li.Network.w c.(i).buf;
          let tmp = (1. /. alpha) *. trace_gemm Mat.(inp *@ c.(i).buf) nci.g in
          predicted_red := !predicted_red +. Maths.(-. alpha +. 0.5 *. sqr alpha) *. tmp
        ) (N.layers net);

      (* possibly update learning rate (Levenberg-Marcquardt-style damping) *)
      let new_loss = N.error (module L) net (x, ytilde) in
      let rho = (new_loss -. loss) /. !predicted_red in
      begin match damping with
        | `Fixed _ -> ()
        | `Levenberg _ ->
          if rho < 0.25 then damp := !damp *. 3. /. 2.;
          if rho > 0.75 then damp := !damp *. 2. /. 3.;
      end;
      (loss, sprintf "damping = %.16f" !damp)
    end
    else (0., "")

end

