make clean
make

perf stat -e branch-instructions,branch-misses,instructions,cpu-cycles,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses,L1-dcache-loads,L1-dcache-load-misses,faults,cs,migrations,cpu-clock ./euler3d ../../data/cfd/fvcorr.domn.097K -t gpu -d 0 -f 0
