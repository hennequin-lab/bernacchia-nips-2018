open Owl
module Mat = Dense.Matrix.S
module Lin = Linalg.S

(* --------------------------------------------------------------------------------
   ---     Some miscellaneous stuff                                             ---
   -------------------------------------------------------------------------------- *)

let to_cblas x =  Bigarray.reshape x [| Mat.numel x |] |> Bigarray.array1_of_genarray

let dotprod x y = Owl_cblas_basic.dot Mat.(numel x) (to_cblas x) 1 (to_cblas y) 1

let ger ?(alpha=1.) a x y =
  let m, n = Mat.shape a in
  Owl_cblas_basic.(ger CblasRowMajor m n alpha 
                     (to_cblas x) 1 
                     (to_cblas y) 1 
                     (to_cblas a) m)

let gemm ?(transa=false) ?(transb=false) ?(alpha=1.) ?(beta=0.) a b c =
  let swap tr (x,y) = if tr then (y, x) else (x, y) in
  let m , n  = Mat.shape c in
  let m_, k  = Mat.shape a |> swap transa in
  let k_, n_ = Mat.shape b |> swap transb in
  assert ((k=k_) && (m=m_) && (n=n_));
  let open Owl_cblas_basic in
  let lda = if transa then m else k in
  let ldb = if transb then k else n in
  let ldc = n in
  let transa = if transa then CblasTrans else CblasNoTrans in 
  let transb = if transb then CblasTrans else CblasNoTrans in 
  gemm CblasRowMajor transa transb m n k 
    alpha (to_cblas a) lda (to_cblas b) ldb
    beta (to_cblas c) ldc

let trace_gemm x y = Mat.(sum' (sum ~axis:1 (x * y)))

let rows a = Bigarray.Genarray.sub_left a 

let last v = v.(Array.length v - 1)

(* extra a minibatch *) 
let get_minibatch set minibatch_size =
  let x, y = set in
  let m, _ = Mat.shape x in
  let ids = Array.init m (fun i -> i) in
  let ids = Stats.choose ids minibatch_size |> Array.to_list in
  let slice = [ L ids; R []] in
  Mat.get_fancy slice x, Mat.get_fancy slice y

(* save error to file *)
let save err filename =
  let err = [| err |> List.rev |> Array.of_list |] |> Mat.of_arrays |> Mat.transpose in
  Mat.save_txt err filename


let cor_noise ?(epsilon=1E-8) ~sigma ~rank n =
  let q, _, _ = Lin.qr Mat.(gaussian n n) in
  let lambdas = Mat.init n 1 (fun i -> epsilon +. exp (-. float i /. (float n /. rank))) in
  let lambdas = Mat.(Maths.(sqr sigma *. float n /. Mat.sum' lambdas) $* lambdas) in
  let c = Mat.(((sqrt lambdas) * q)) in
  fun minibatch_size -> Mat.((gaussian minibatch_size n) *@ c)


let time_this f =
  let t_start = Unix.gettimeofday () in
  let result = f () in
  let t_end = Unix.gettimeofday () in
  result, t_end -. t_start

let truncated_svd ~tol a =
  let _, s, vt = Lin.svd ~thin:true a in
  let rank = Array.length (Mat.filter (fun s -> s >= tol) s) in
  let slice = [0; rank-1] in
  let s = Mat.get_slice [[]; slice] s in
  let vt = Mat.get_slice [slice; []] vt in
  s, vt

