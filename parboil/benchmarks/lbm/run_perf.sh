cd /home/pacman/zf/parboil/parboil/benchmarks/lbm
perf stat -e branch-instructions,branch-misses,instructions,cpu-cycles,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses,L1-dcache-loads,L1-dcache-load-misses,faults,cs,migrations,cpu-clock ./build/opencl_base_default/lbm -p 100 -i /home/pacman/zf/parboil/parboil/datasets/lbm/short/input/120_120_150_ldc.of -o /home/pacman/zf/parboil/parboil/benchmarks/lbm/run/short/reference.dat -- 100
cp tools/compare-output /home/pacman/zf/parboil/parboil/compare_zf/
cd /home/pacman/zf/parboil/parboil/compare_zf
./compare-output ~/zf/parboil/parboil/benchmarks/lbm/run/short/reference.dat ../datasets/lbm/short/output/reference.dat 

