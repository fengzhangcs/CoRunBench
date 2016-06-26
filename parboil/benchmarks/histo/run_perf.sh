cd /home/pacman/zf/parboil/parboil/benchmarks/histo
perf stat -e branch-instructions,branch-misses,instructions,cpu-cycles,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses,L1-dcache-loads,L1-dcache-load-misses,faults,cs,migrations,cpu-clock ./build/opencl_base_default/histo -p  100 -i /home/pacman/zf/parboil/parboil/datasets/histo/default/input/img.bin -o /home/pacman/zf/parboil/parboil/benchmarks/histo/run/default/ref.bmp -- 20 4
cp tools/compare-output /home/pacman/zf/parboil/parboil/compare_zf/
cd /home/pacman/zf/parboil/parboil/compare_zf
 ./compare-output ~/zf/parboil/parboil/benchmarks/histo/run/default/ref.bmp ../datasets/histo/default/output/ref.bmp 

