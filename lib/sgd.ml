open Printf
open Bigarray
open Owl
open Util

module Make (P: sig
    module N: Network.T
    module L: Loss.T
    val alpha: float
  end) = struct

  open P
  type net = N.t

  type cache = { buf: Mat.mat }

  let cache =
    let net = N.make_random () in
    Array.map (fun l ->
        let a, b = Mat.shape l.Network.w in
        { buf = Mat.zeros a b }
      ) (N.layers net)

  let step ~t net ~update (x, ytilde) =
    let data_size, _ = Mat.shape x in
    let loss = N.forward_backward (module L) net (x, ytilde) in
    let nc, _ = N.get_cache net (Mat.row_num x) in
    Array.iteri (fun i li ->
        let ci = cache.(i) in
        let nci = nc.(i) in
        gemm ~transa:true nci.inp nci.g ci.buf;
        Mat.scalar_mul_ (alpha /. float data_size) ci.buf;
        if update then Mat.sub_ li.Network.w ci.buf
      ) (N.layers net);
    (loss, "")

end

