
cd /home/pacman/zf/parboil/parboil/benchmarks/mri-gridding
perf stat -e branch-instructions,branch-misses,instructions,cpu-cycles,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses,L1-dcache-loads,L1-dcache-load-misses,faults,cs,migrations,cpu-clock ./build/opencl_base_default/mri-gridding -p 100 -i /home/pacman/zf/parboil/parboil/datasets/mri-gridding/small/input/small.uks -o /home/pacman/zf/parboil/parboil/benchmarks/mri-gridding/run/small/output.txt
cp tools/compare-output /home/pacman/zf/parboil/parboil/compare_zf/
cd /home/pacman/zf/parboil/parboil/compare_zf
./compare-output ~/zf/parboil/parboil/benchmarks/mri-gridding/run/small/output.txt ../datasets/mri-gridding/small/output/output.txt 


