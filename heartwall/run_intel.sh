#!/bin/bash
rm result*
make clean
make
Start="4" 
#for j in 0
#do
for i in 0 10 20 30 40 50 60 70 80 90 100
#for i in 0 1 2 4 8 16 32 50 64 75 100
#for i in 100 99 98 96 92 84 68 50 36 25 0
do
 #while [ $Start -le 20 ]
# do
   echo "####cpu_work: $i% "
	./a.out -no_frames "$Start" -cpu_offset "$i" -device "$j" >> result.txt
	#echo $out
  #  out=./a.out "-file" "input/mil.txt" "-cpu_offset" "$i"
#	 Start=$(($Start + 1))
done
#done
#done
