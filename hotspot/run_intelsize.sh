#!/bin/bash
rm perf*
rm result*
make clean
make
#Start = "1"
for j in 0
do
for i in 0 10 20 30 40 50 60 70 80 90 100
#for i in 0 1 2 4 8 16 32 50 64 75 100
#for i in 100 99 98 96 92 84 68 50 36 25 0
do
 #while [ $Start -le 10 ]
 #do
   echo "cpu_work: $i% "
	./hotspot 512 2 1000 ../../data/hotspot/temp_512 ../../data/hotspot/power_512 output.out "$i" "$j" >>result.txt
#done
done
done
