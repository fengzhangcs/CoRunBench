make clean
#make
make KERNEL_DIM="-DRD_WG_SIZE_0=64 -DRD_WG_SIZE_1_0=64 -DRD_WG_SIZE_1_1=1"
./gaussian -s 2048 -a 0
./gaussian -s 2048 -a 10
./gaussian -s 2048 -a 20
./gaussian -s 2048 -a 30
./gaussian -s 2048 -a 40
./gaussian -s 2048 -a 50
./gaussian -s 2048 -a 60
./gaussian -s 2048 -a 70
./gaussian -s 2048 -a 80
./gaussian -s 2048 -a 90
./gaussian -s 2048 -a 100
