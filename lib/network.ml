open Printf
open Bigarray
open Owl
open Util

(* for layer i... *)
type layer = {
  w: Mat.mat; (* feedforward weights into layer i, including biases *)
  f: Mat.mat -> unit; (* activation function, *in place modif* *)
  fprime: Mat.mat -> unit; (* derivative of activation function, *in place modif* *)
}

type layer_cache = {
  inp: Mat.mat; (* abar of previous layer *)
  s: Mat.mat; (* potential *)
  a: Mat.mat; (* activations *)
  abar: Mat.mat; (* activation, appending 1.0 at the end *)
  da: Mat.mat; (* derivative of activation function *)
  g: Mat.mat; (* adjoint *)   
}

(* --------------------------------------------------------------------------------
   ---     Main network module                                                  ---
   -------------------------------------------------------------------------------- *)

module type T = sig
  type t
  val input_size: int
  val n_layers: t -> int
  val n_prms: t -> int
  val layers: t -> layer array
  val get_cache: t -> int -> (layer_cache array * Mat.mat)
  val make_random: ?sigmas:float array -> ?b:float -> unit -> t
  val copy: t -> t
  val copy_into: t -> t -> unit
  val save: t -> (string -> string) -> unit
  val error: (module Loss.T) -> t -> Mat.mat * Mat.mat -> float
  val run: t -> Mat.mat -> Mat.mat
  val forward_pass: t -> Mat.mat -> unit
  val backward_pass: t -> int -> unit
  val forward_backward : (module Loss.T) -> t -> Mat.mat * Mat.mat -> float
end

module Make (P: sig
    val input_size: int
    val layer_sizes: int array
    val act_funs: ((Mat.mat -> unit) * (Mat.mat -> unit)) array
  end) : T = struct

  open P
  let input_size = input_size

  type t = { layers: layer array;
             cache: (int, layer_cache array * Mat.mat) Hashtbl.t }

  let n_layers net = Array.length net.layers
  let n_prms net = Array.fold_left (fun accu l -> accu + Mat.numel l.w) 0 net.layers
  let layers net = net.layers

  let get_cache net data_size =
    try Hashtbl.find net.cache data_size 
    with Not_found -> 
      let c = Array.map (fun l -> 
          let m, n = Mat.shape l.w in
          let inp = Mat.empty data_size m in
          let s = Mat.zeros data_size n in
          let a = Mat.zeros data_size n in
          let abar = Mat.(concat_horizontal (zeros data_size n) (ones data_size 1)) in
          let da = Mat.zeros data_size n in
          let g = Mat.zeros data_size n in
          { inp; s; abar; a; da; g }
        ) net.layers in
      let loss_buf = Mat.copy (last c).a in
      Hashtbl.add net.cache data_size (c, loss_buf);
      c, loss_buf

  let copy net =
    let layers = Array.map (fun l -> {l with w = Mat.copy l.w }) net.layers in
    { layers; cache = Hashtbl.create 1 }

  let copy_into src dst =
    for i=0 to pred (n_layers src) do
      Genarray.blit src.layers.(i).w dst.layers.(i).w  
    done

  let save net in_dir = Array.iteri (fun i li ->
      Mat.(save_txt (transpose li.w) (in_dir (sprintf "w%i" i)))
    ) net.layers

  let make_random ?sigmas ?(b=0.) () =
    let layers = Array.mapi (fun i n ->
        let m = if i=0 then input_size else layer_sizes.(pred i) in
        let f, fprime = act_funs.(i) in 
        let sigma = match sigmas with
          | Some s -> s.(i)
          | None -> 1. in
        let w = 
          let weights = Mat.gaussian ~sigma:(sigma /. sqrt (float m)) m n in
          let biases = Mat.create 1 n b in
          Mat.concat_vertical weights biases in
        { w; f; fprime }
      ) layer_sizes in
    let cache = Hashtbl.create 1 in
    { layers; cache } 


  let forward_pass net x =
    let data_size, _ = Mat.shape x in
    let cache, _ = get_cache net data_size in
    let _ =  (* fill first inp *)
      let first = cache.(0) in
      for k=0 to pred data_size do
        for j=0 to pred input_size do Mat.set first.inp k j (Mat.get x k j) done;
        Mat.set first.inp k input_size 1.
      done in
    Array.iteri (fun i li ->
        let _, n = Mat.shape li.w in
        let ci = cache.(i) in
        if i>0 then Genarray.blit cache.(i-1).abar ci.inp;
        gemm ci.inp li.w ci.s;
        Genarray.blit ci.s ci.a;
        li.f ci.a;
        for k=0 to pred data_size do
          for j=0 to n-1 do Mat.set ci.abar k j (Mat.get ci.a k j) done;
          Mat.set ci.abar k n 1.
        done;
      ) net.layers

  (* assumes the forward pass has just been run *)
  let backward_pass net data_size =
    let cache, _ = get_cache net data_size in
    for i=pred (n_layers net) downto 0 do
      let li = net.layers.(i) in
      let ci = cache.(i) in
      Genarray.blit ci.s ci.g;
      li.fprime ci.g;
      Mat.mul_ ci.g ci.da;
      let w_trunc = rows li.w 0 (Mat.(row_num li.w) - 1) in
      if i>0 then gemm ~transb:true ci.g w_trunc cache.(pred i).da;
    done

  let run net x =
    let data_size, _ = Mat.shape x in
    let cache, _ = get_cache net data_size in
    forward_pass net x;
    Mat.copy (last cache).a

  let error (module L: Loss.T) net (x, ytilde) =
    let data_size, _ = Mat.shape x in
    let cache, loss_buf = get_cache net data_size in
    forward_pass net x;
    let y = (last cache).a in
    L.loss ~buf:loss_buf ~y ~ytilde

  let forward_backward (module L: Loss.T) net (x, ytilde) =
    let data_size, _ = Mat.shape x in
    let cache, loss_buf = get_cache net data_size in
    forward_pass net x;
    let y = (last cache).a in
    let loss = L.loss ~buf:loss_buf ~y ~ytilde in
    L.dloss ~into:(last cache).da ~y ~ytilde;
    backward_pass net data_size;
    loss

end

(* --------------------------------------------------------------------------------
   ---     Useful activation functions                                          ---
   -------------------------------------------------------------------------------- *)

let linear = (fun _ -> ()), (fun x -> Mat.fill x 1.)
let relu = (fun x -> Mat.relu_ x), (fun x -> Mat.signum_ x; Mat.add_scalar_ x 1.; Mat.mul_scalar_ x 0.5)
let tanh = (fun x -> Mat.tanh_ x), (fun x -> Mat.tanh_ x; Mat.sqr_ x; Mat.mul_scalar_ x (-1.); Mat.add_scalar_ x 1.)


