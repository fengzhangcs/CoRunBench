cd /home/pacman/zf/parboil/parboil/benchmarks/sgemm
./build/opencl_base_default/sgemm -p 0 -i /home/pacman/zf/parboil/parboil/datasets/sgemm/medium/input/matrix1.txt,/home/pacman/zf/parboil/parboil/datasets/sgemm/medium/input/matrix2t.txt,/home/pacman/zf/parboil/parboil/datasets/sgemm/medium/input/matrix2t.txt -o /home/pacman/zf/parboil/parboil/benchmarks/sgemm/run/medium/matrix3.txt
cp tools/compare-output /home/pacman/zf/parboil/parboil/compare_zf/
cd /home/pacman/zf/parboil/parboil/compare_zf
./compare-output ~/zf/parboil/parboil/benchmarks/sgemm/run/medium/matrix3.txt ../datasets/sgemm/medium/output/matrix3.txt 

cd /home/pacman/zf/parboil/parboil/benchmarks/sgemm
./build/opencl_base_default/sgemm -p 10 -i /home/pacman/zf/parboil/parboil/datasets/sgemm/medium/input/matrix1.txt,/home/pacman/zf/parboil/parboil/datasets/sgemm/medium/input/matrix2t.txt,/home/pacman/zf/parboil/parboil/datasets/sgemm/medium/input/matrix2t.txt -o /home/pacman/zf/parboil/parboil/benchmarks/sgemm/run/medium/matrix3.txt
cp tools/compare-output /home/pacman/zf/parboil/parboil/compare_zf/
cd /home/pacman/zf/parboil/parboil/compare_zf
./compare-output ~/zf/parboil/parboil/benchmarks/sgemm/run/medium/matrix3.txt ../datasets/sgemm/medium/output/matrix3.txt 

cd /home/pacman/zf/parboil/parboil/benchmarks/sgemm
./build/opencl_base_default/sgemm -p 20 -i /home/pacman/zf/parboil/parboil/datasets/sgemm/medium/input/matrix1.txt,/home/pacman/zf/parboil/parboil/datasets/sgemm/medium/input/matrix2t.txt,/home/pacman/zf/parboil/parboil/datasets/sgemm/medium/input/matrix2t.txt -o /home/pacman/zf/parboil/parboil/benchmarks/sgemm/run/medium/matrix3.txt
cp tools/compare-output /home/pacman/zf/parboil/parboil/compare_zf/
cd /home/pacman/zf/parboil/parboil/compare_zf
./compare-output ~/zf/parboil/parboil/benchmarks/sgemm/run/medium/matrix3.txt ../datasets/sgemm/medium/output/matrix3.txt 

cd /home/pacman/zf/parboil/parboil/benchmarks/sgemm
./build/opencl_base_default/sgemm -p 30 -i /home/pacman/zf/parboil/parboil/datasets/sgemm/medium/input/matrix1.txt,/home/pacman/zf/parboil/parboil/datasets/sgemm/medium/input/matrix2t.txt,/home/pacman/zf/parboil/parboil/datasets/sgemm/medium/input/matrix2t.txt -o /home/pacman/zf/parboil/parboil/benchmarks/sgemm/run/medium/matrix3.txt
cp tools/compare-output /home/pacman/zf/parboil/parboil/compare_zf/
cd /home/pacman/zf/parboil/parboil/compare_zf
./compare-output ~/zf/parboil/parboil/benchmarks/sgemm/run/medium/matrix3.txt ../datasets/sgemm/medium/output/matrix3.txt 

cd /home/pacman/zf/parboil/parboil/benchmarks/sgemm
./build/opencl_base_default/sgemm -p 40 -i /home/pacman/zf/parboil/parboil/datasets/sgemm/medium/input/matrix1.txt,/home/pacman/zf/parboil/parboil/datasets/sgemm/medium/input/matrix2t.txt,/home/pacman/zf/parboil/parboil/datasets/sgemm/medium/input/matrix2t.txt -o /home/pacman/zf/parboil/parboil/benchmarks/sgemm/run/medium/matrix3.txt
cp tools/compare-output /home/pacman/zf/parboil/parboil/compare_zf/
cd /home/pacman/zf/parboil/parboil/compare_zf
./compare-output ~/zf/parboil/parboil/benchmarks/sgemm/run/medium/matrix3.txt ../datasets/sgemm/medium/output/matrix3.txt 

cd /home/pacman/zf/parboil/parboil/benchmarks/sgemm
./build/opencl_base_default/sgemm -p 50 -i /home/pacman/zf/parboil/parboil/datasets/sgemm/medium/input/matrix1.txt,/home/pacman/zf/parboil/parboil/datasets/sgemm/medium/input/matrix2t.txt,/home/pacman/zf/parboil/parboil/datasets/sgemm/medium/input/matrix2t.txt -o /home/pacman/zf/parboil/parboil/benchmarks/sgemm/run/medium/matrix3.txt
cp tools/compare-output /home/pacman/zf/parboil/parboil/compare_zf/
cd /home/pacman/zf/parboil/parboil/compare_zf
./compare-output ~/zf/parboil/parboil/benchmarks/sgemm/run/medium/matrix3.txt ../datasets/sgemm/medium/output/matrix3.txt 

cd /home/pacman/zf/parboil/parboil/benchmarks/sgemm
./build/opencl_base_default/sgemm -p 60 -i /home/pacman/zf/parboil/parboil/datasets/sgemm/medium/input/matrix1.txt,/home/pacman/zf/parboil/parboil/datasets/sgemm/medium/input/matrix2t.txt,/home/pacman/zf/parboil/parboil/datasets/sgemm/medium/input/matrix2t.txt -o /home/pacman/zf/parboil/parboil/benchmarks/sgemm/run/medium/matrix3.txt
cp tools/compare-output /home/pacman/zf/parboil/parboil/compare_zf/
cd /home/pacman/zf/parboil/parboil/compare_zf
./compare-output ~/zf/parboil/parboil/benchmarks/sgemm/run/medium/matrix3.txt ../datasets/sgemm/medium/output/matrix3.txt 

cd /home/pacman/zf/parboil/parboil/benchmarks/sgemm
./build/opencl_base_default/sgemm -p 70 -i /home/pacman/zf/parboil/parboil/datasets/sgemm/medium/input/matrix1.txt,/home/pacman/zf/parboil/parboil/datasets/sgemm/medium/input/matrix2t.txt,/home/pacman/zf/parboil/parboil/datasets/sgemm/medium/input/matrix2t.txt -o /home/pacman/zf/parboil/parboil/benchmarks/sgemm/run/medium/matrix3.txt
cp tools/compare-output /home/pacman/zf/parboil/parboil/compare_zf/
cd /home/pacman/zf/parboil/parboil/compare_zf
./compare-output ~/zf/parboil/parboil/benchmarks/sgemm/run/medium/matrix3.txt ../datasets/sgemm/medium/output/matrix3.txt 

cd /home/pacman/zf/parboil/parboil/benchmarks/sgemm
./build/opencl_base_default/sgemm -p 80 -i /home/pacman/zf/parboil/parboil/datasets/sgemm/medium/input/matrix1.txt,/home/pacman/zf/parboil/parboil/datasets/sgemm/medium/input/matrix2t.txt,/home/pacman/zf/parboil/parboil/datasets/sgemm/medium/input/matrix2t.txt -o /home/pacman/zf/parboil/parboil/benchmarks/sgemm/run/medium/matrix3.txt
cp tools/compare-output /home/pacman/zf/parboil/parboil/compare_zf/
cd /home/pacman/zf/parboil/parboil/compare_zf
./compare-output ~/zf/parboil/parboil/benchmarks/sgemm/run/medium/matrix3.txt ../datasets/sgemm/medium/output/matrix3.txt 

cd /home/pacman/zf/parboil/parboil/benchmarks/sgemm
./build/opencl_base_default/sgemm -p 90 -i /home/pacman/zf/parboil/parboil/datasets/sgemm/medium/input/matrix1.txt,/home/pacman/zf/parboil/parboil/datasets/sgemm/medium/input/matrix2t.txt,/home/pacman/zf/parboil/parboil/datasets/sgemm/medium/input/matrix2t.txt -o /home/pacman/zf/parboil/parboil/benchmarks/sgemm/run/medium/matrix3.txt
cp tools/compare-output /home/pacman/zf/parboil/parboil/compare_zf/
cd /home/pacman/zf/parboil/parboil/compare_zf
./compare-output ~/zf/parboil/parboil/benchmarks/sgemm/run/medium/matrix3.txt ../datasets/sgemm/medium/output/matrix3.txt 

cd /home/pacman/zf/parboil/parboil/benchmarks/sgemm
./build/opencl_base_default/sgemm -p 100 -i /home/pacman/zf/parboil/parboil/datasets/sgemm/medium/input/matrix1.txt,/home/pacman/zf/parboil/parboil/datasets/sgemm/medium/input/matrix2t.txt,/home/pacman/zf/parboil/parboil/datasets/sgemm/medium/input/matrix2t.txt -o /home/pacman/zf/parboil/parboil/benchmarks/sgemm/run/medium/matrix3.txt
cp tools/compare-output /home/pacman/zf/parboil/parboil/compare_zf/
cd /home/pacman/zf/parboil/parboil/compare_zf
./compare-output ~/zf/parboil/parboil/benchmarks/sgemm/run/medium/matrix3.txt ../datasets/sgemm/medium/output/matrix3.txt 

