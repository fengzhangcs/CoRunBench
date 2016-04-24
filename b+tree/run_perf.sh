#!/bin/bash
#for j in 0 1 2 
rm perf*
rm result*


sed -i '1s/^.*$/#define DEFAULT_ORDER 256/' define.h
cp define.h /tmp/define.h
make clean
make	

perf stat -e branch-instructions,branch-misses,instructions,cpu-cycles,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses,L1-dcache-loads,L1-dcache-load-misses,faults,cs,migrations,cpu-clock ./a.out -file "input/mil.txt" -cpu_offset "0"  -device 0 -commandJ 65536 "10000"


#Total time is the GPU: KERNEL time, tony runs it 3 times and get the average
