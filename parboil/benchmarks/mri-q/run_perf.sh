cd /home/pacman/zf/parboil/parboil/benchmarks/mri-q
perf stat -e branch-instructions,branch-misses,instructions,cpu-cycles,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses,L1-dcache-loads,L1-dcache-load-misses,faults,cs,migrations,cpu-clock ./build/opencl_default/mri-q -p 100 -i /home/pacman/zf/parboil/parboil/datasets/mri-q/large/input/64_64_64_dataset.bin -o /home/pacman/zf/parboil/parboil/benchmarks/mri-q/run/large/64_64_64_dataset.out
cp tools/compare-output /home/pacman/zf/parboil/parboil/compare_zf/
cd /home/pacman/zf/parboil/parboil/compare_zf
./compare-output ~/zf/parboil/parboil/benchmarks/mri-q/run/large/64_64_64_dataset.out ../datasets/mri-q/large/output/64_64_64_dataset.out 


