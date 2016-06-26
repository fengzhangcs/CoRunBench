
cd /home/pacman/zf/parboil/parboil/benchmarks/spmv
perf stat -e branch-instructions,branch-misses,instructions,cpu-cycles,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses,L1-dcache-loads,L1-dcache-load-misses,faults,cs,migrations,cpu-clock ./build/opencl_base_default/spmv -p 100 -i /home/pacman/zf/parboil/parboil/datasets/spmv/large/input/Dubcova3.mtx.bin,/home/pacman/zf/parboil/parboil/datasets/spmv/large/input/vector.bin -o /home/pacman/zf/parboil/parboil/benchmarks/spmv/run/large/Dubcova3.mtx.out
cp tools/compare-output /home/pacman/zf/parboil/parboil/compare_zf/
cd /home/pacman/zf/parboil/parboil/compare_zf
./compare-output ~/zf/parboil/parboil/benchmarks/spmv/run/large/Dubcova3.mtx.out ../datasets/spmv/large/output/Dubcova3.mtx.out


