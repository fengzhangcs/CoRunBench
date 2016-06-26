
cd /home/pacman/zf/parboil/parboil/benchmarks/sad
perf stat -e branch-instructions,branch-misses,instructions,cpu-cycles,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses,L1-dcache-loads,L1-dcache-load-misses,faults,cs,migrations,cpu-clock ./build/opencl_base_default/sad -p 100 -i /home/pacman/zf/parboil/parboil/datasets/sad/default/input/reference.bin,/home/pacman/zf/parboil/parboil/datasets/sad/default/input/frame.bin -o /home/pacman/zf/parboil/parboil/benchmarks/sad/run/default/out.bin
cp tools/compare-output /home/pacman/zf/parboil/parboil/compare_zf/
cd /home/pacman/zf/parboil/parboil/compare_zf
./compare-output ~/zf/parboil/parboil/benchmarks/sad/run/default/out.bin ../datasets/sad/default/output/out.bin 


