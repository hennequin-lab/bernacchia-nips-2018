module type T = sig
  type net
  val step: t:int 
    -> net 
    -> update:bool
    -> (Util.Mat.mat * Util.Mat.mat)
    -> float * string
end
