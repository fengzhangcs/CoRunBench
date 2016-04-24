#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "backprop.h"
#include "omp.h"
#include <sys/time.h>

extern char *strcpy();
extern void exit();
extern int cpu_offset;

int layer_size = 0;

double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

void backprop_face()
{
  BPNN *net;
  int i;
  float out_err, hid_err;
  net = bpnn_create(layer_size, 16, 1); // (16, 1 can not be changed)
  
  printf("Input layer size : %d\n", layer_size);
  load(net);
  //entering the training kernel, only one iteration
  printf("Starting training kernel\n");
  double starttime = gettime();
  bpnn_train_kernel(net, &out_err, &hid_err);
  double endtime = gettime();
   printf("CAUTION: cpu_offset: %d time: %lf mseconds\n", cpu_offset, 1000*(endtime-starttime));
  bpnn_free(net);
  printf("\nFinish the training for one iteration\n");
}

int setup(int argc, char **argv)
{
	
  int seed;

  if (argc!=3){
  fprintf(stderr, "usage: backprop <num of input elements> cpu_offset\n");
  exit(0);
  }
  cpu_offset=atoi(argv[2]);//zf
  printf("cpuoffset = %d\n", cpu_offset);
  layer_size = atoi(argv[1]);
  if (layer_size%16!=0){
  fprintf(stderr, "The number of input points must be divided by 16\n");
  exit(0);
  }
  

  seed = 7;   
  bpnn_initialize(seed);
  backprop_face();

  exit(0);
}
