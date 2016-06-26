cd /home/pacman/zf/parboil/parboil/benchmarks/sgemm
perf stat -e branch-instructions,branch-misses,instructions,cpu-cycles,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses,L1-dcache-loads,L1-dcache-load-misses,faults,cs,migrations,cpu-clock ./build/opencl_base_default/sgemm -p 100 -i /home/pacman/zf/parboil/parboil/datasets/sgemm/medium/input/matrix1.txt,/home/pacman/zf/parboil/parboil/datasets/sgemm/medium/input/matrix2t.txt,/home/pacman/zf/parboil/parboil/datasets/sgemm/medium/input/matrix2t.txt -o /home/pacman/zf/parboil/parboil/benchmarks/sgemm/run/medium/matrix3.txt
cp tools/compare-output /home/pacman/zf/parboil/parboil/compare_zf/
cd /home/pacman/zf/parboil/parboil/compare_zf
./compare-output ~/zf/parboil/parboil/benchmarks/sgemm/run/medium/matrix3.txt ../datasets/sgemm/medium/output/matrix3.txt 

