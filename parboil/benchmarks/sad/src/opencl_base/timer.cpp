#include<sys/time.h>
#include<stdio.h>
double gettime() {
    struct timeval t;
      gettimeofday(&t,NULL);
        return t.tv_sec+t.tv_usec*1e-6;
}

