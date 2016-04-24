// EXAMPLE:
  35 // ./a.out -file ./input/mil.txt
  36 // ...then enter any of the following commands after the prompt > :
  37 // f <x>  -- Find the value under key <x>
  38 // p <x> -- Print the path from the root to key k and its associated value
  39 // t -- Print the B+ tree
  40 // l -- Print the keys of the leaves (bottom row of the tree)
  41 // v -- Toggle output of pointer addresses ("verbose") in tree and leaves.
  42 // k <x> -- [uses GPU] Run <x> bundled queries on the CPU and GPU (B+Tree) (Selects random values for each search)
  43 // j <x> <y> -- [uses GPU] Run a range search of <x> bundled queries on the CPU and GPU (B+Tree) with the range of each search of size <y>
  44 // x <z> -- Run a single search for value z on the GPU and CPU
  45 // y <a> <b> -- Run a single range search for range a-b on the GPU and CPU
  46 // q -- Quit. (Or use Ctl-D.) 

