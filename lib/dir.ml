let dir = Cmdargs.(get_string "-d" |> force ~usage:"-d directory")
let in_dir = Printf.sprintf "%s/%s" dir
