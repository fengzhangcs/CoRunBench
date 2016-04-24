#!/bin/bash
rm result*
make clean
make
Start="20" 
perf stat -e branch-instructions,branch-misses,instructions,cpu-cycles,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses,L1-dcache-loads,L1-dcache-load-misses,faults,cs,migrations,cpu-clock ./a.out -no_frames "$Start" -cpu_offset "0" -device "0" 
