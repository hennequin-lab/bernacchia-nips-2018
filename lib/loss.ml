open Bigarray
open Owl
open Util

module type T = sig
  val loss: buf:Mat.mat -> y:Mat.mat -> ytilde:Mat.mat -> float
  val dloss: into:Mat.mat -> y:Mat.mat -> ytilde:Mat.mat -> unit
  val sample: Mat.mat -> Mat.mat
end

module Squared : T = struct

  let loss ~buf ~y ~ytilde =
    Genarray.blit y buf; 
    Mat.sub_ buf ytilde; 
    0.5 *. Mat.l2norm_sqr' buf /. float Mat.(row_num y)

  let dloss ~into ~y ~ytilde = 
    Genarray.blit y into; 
    Mat.sub_ into ytilde;
    Mat.scalar_mul_ (1. /. float Mat.(row_num y)) into

  let sample y = Mat.(y + gaussian (row_num y) (col_num y))

end


module Cross_entropy : T = struct

  let loss ~buf ~y ~ytilde =
    Genarray.blit y buf;
    Mat.softmax_ ~axis:1 buf;
    Mat.(cross_entropy' ytilde buf) /. float Mat.(row_num y)

  let dloss ~into ~y ~ytilde = 
    Genarray.blit y into; 
    Mat.sub_ into ytilde;
    Mat.scalar_mul_ (1. /. float Mat.(row_num y)) into

  let sample y = 
    let a, b = Mat.shape y in
    let dists = Mat.(softmax ~axis:1 y) in
    let samples = Mat.zeros a b in
    Array.iteri (fun i dist ->
        let k = if Random.float 1. < 0.5 then Random.int 10 else Stats.categorical_rvs dist in
        Mat.set samples i k 1.
      ) (Mat.to_arrays dists);
    samples

end


