#!/bin/bash
#for j in 0 1 2 
rm perf*
rm result*


sed -i '1s/^.*$/#define DEFAULT_ORDER 256/' define.h
cp define.h /tmp/define.h
make clean
make	

./a.out -file "input/mil.txt" -cpu_offset "0"  -device 0 -commandJ 65536 "10000"
./a.out -file "input/mil.txt" -cpu_offset "10"  -device 0 -commandJ 65536 "10000"
./a.out -file "input/mil.txt" -cpu_offset "20"  -device 0 -commandJ 65536 "10000"
./a.out -file "input/mil.txt" -cpu_offset "30"  -device 0 -commandJ 65536 "10000"
./a.out -file "input/mil.txt" -cpu_offset "40"  -device 0 -commandJ 65536 "10000"
./a.out -file "input/mil.txt" -cpu_offset "50"  -device 0 -commandJ 65536 "10000"
./a.out -file "input/mil.txt" -cpu_offset "60"  -device 0 -commandJ 65536 "10000"
./a.out -file "input/mil.txt" -cpu_offset "70"  -device 0 -commandJ 65536 "10000" 
./a.out -file "input/mil.txt" -cpu_offset "80"  -device 0 -commandJ 65536 "10000" 
./a.out -file "input/mil.txt" -cpu_offset "90"  -device 0 -commandJ 65536 "10000" 
./a.out -file "input/mil.txt" -cpu_offset "100"  -device 0 -commandJ 65536 "10000" 


#Total time is the GPU: KERNEL time, tony runs it 3 times and get the average
