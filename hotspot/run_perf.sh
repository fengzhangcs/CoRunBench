#!/bin/bash
rm perf*
rm result*
make clean
make

perf stat -e branch-instructions,branch-misses,instructions,cpu-cycles,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses,L1-dcache-loads,L1-dcache-load-misses,faults,cs,migrations,cpu-clock	./hotspot 512 2 1000 ../../data/hotspot/temp_512 ../../data/hotspot/power_512 output.out "0" "0" 


