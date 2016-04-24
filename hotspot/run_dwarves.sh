#!/bin/bash
rm perf*
rm result*
make clean
make

	./hotspot 512 2 1000 ../../data/hotspot/temp_512 ../../data/hotspot/power_512 output.out "0" "0" 
	./hotspot 512 2 1000 ../../data/hotspot/temp_512 ../../data/hotspot/power_512 output.out "10" "0" 
	./hotspot 512 2 1000 ../../data/hotspot/temp_512 ../../data/hotspot/power_512 output.out "20" "0" 
	./hotspot 512 2 1000 ../../data/hotspot/temp_512 ../../data/hotspot/power_512 output.out "30" "0" 
	./hotspot 512 2 1000 ../../data/hotspot/temp_512 ../../data/hotspot/power_512 output.out "40" "0" 
	./hotspot 512 2 1000 ../../data/hotspot/temp_512 ../../data/hotspot/power_512 output.out "50" "0" 
	./hotspot 512 2 1000 ../../data/hotspot/temp_512 ../../data/hotspot/power_512 output.out "60" "0" 
	./hotspot 512 2 1000 ../../data/hotspot/temp_512 ../../data/hotspot/power_512 output.out "70" "0" 
	./hotspot 512 2 1000 ../../data/hotspot/temp_512 ../../data/hotspot/power_512 output.out "80" "0" 
	./hotspot 512 2 1000 ../../data/hotspot/temp_512 ../../data/hotspot/power_512 output.out "90" "0" 
	./hotspot 512 2 1000 ../../data/hotspot/temp_512 ../../data/hotspot/power_512 output.out "100" "0" 


