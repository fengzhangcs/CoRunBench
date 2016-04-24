#!/bin/bash
rm perf*
rm result*
make clean
make 
for j in 0 1 2
do
for i in 0 1 2 4 8 16 32 50 64 75 100
do
   echo "cpu_work: $i% "
	perf stat -o perf.txt --append -e cycles,instructions,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses  ./a.out -no_frames "20" -cpu_offset "$i" -device "$j" >> result.txt
	#echo $out
  #  out=./a.out "-file" "input/mil.txt" "-cpu_offset" "$i"
done
done
