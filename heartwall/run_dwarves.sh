#!/bin/bash
rm result*
make clean
make
Start="20" 
./a.out -no_frames "$Start" -cpu_offset "0" -device "0" 
./a.out -no_frames "$Start" -cpu_offset "10" -device "0" 
./a.out -no_frames "$Start" -cpu_offset "20" -device "0" 
./a.out -no_frames "$Start" -cpu_offset "30" -device "0" 
./a.out -no_frames "$Start" -cpu_offset "40" -device "0" 
./a.out -no_frames "$Start" -cpu_offset "50" -device "0" 
./a.out -no_frames "$Start" -cpu_offset "60" -device "0" 
./a.out -no_frames "$Start" -cpu_offset "70" -device "0" 
./a.out -no_frames "$Start" -cpu_offset "80" -device "0" 
./a.out -no_frames "$Start" -cpu_offset "90" -device "0" 
./a.out -no_frames "$Start" -cpu_offset "100" -device "0" 
