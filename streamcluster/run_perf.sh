make clean
make
perf stat -e branch-instructions,branch-misses,instructions,cpu-cycles,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses,L1-dcache-loads,L1-dcache-load-misses,faults,cs,migrations,cpu-clock ./streamcluster 10 20 256 65536 65536 1000 none output.txt 1 -t gpu -d 0 -f 100
