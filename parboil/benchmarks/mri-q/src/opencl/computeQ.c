/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include <stdio.h>
#include <malloc.h>
#include <CL/cl.h>
#include "ocl.h"
#include "macros.h"
#include "computeQ.h"

extern int cpu_offset ;


#define NC 4

cl_int clEnqueueNDRangeKernel_fusion ( cl_command_queue* command_queue,
    cl_kernel kernel,
    cl_uint work_dim,
    const size_t *global_work_offset,
    const size_t *global_work_size,
    const size_t *local_work_size,
    cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list,
    cl_event *event){

  cl_event eventList[2];
  int cpu_run=0, gpu_run=0;
 size_t global_offset[2];
 size_t global_offset_start[2];
 size_t remain_global_work_size[2];
 int i;
 cl_int errcode;


  if(cpu_offset == 0){
    gpu_run=1;
  }
  else if(cpu_offset == 100){
    cpu_run=1;
  }
  else{
    gpu_run=1;
    cpu_run=1;
  }

    for(i=0; i<work_dim; i++){
    global_offset[i]=global_work_size[i];   
    remain_global_work_size[i]=global_work_size[i];   
  }

  global_offset[0]=((double)cpu_offset/100)*global_work_size[0];   
  int t1=global_offset[0], t2=local_work_size[0];
  global_offset[0]=(t1/t2+(int)(t1%t2!=0))*t2;
//  printf("global_work_size[0]=%d, global_offset[0]=%d\n",global_work_size[0],global_offset[0]);
  remain_global_work_size[0] = global_work_size[0]-global_offset[0];
  if(remain_global_work_size[0] == 0)
    gpu_run = 0;
  if(global_offset[0] == 0)
    cpu_run = 0;

  global_offset_start[0]=global_offset[0];
  global_offset_start[1]=0;
 
  if(gpu_run){
    errcode = clEnqueueNDRangeKernel(command_queue[0], kernel, work_dim, global_offset_start, remain_global_work_size, local_work_size, 0, NULL, &(eventList[0]));
    if(errcode != CL_SUCCESS) printf("Error in gpu clEnqueueNDRangeKernel\n");
  }
//  clFinish(command_queue[0]);
  if(cpu_run){
//    printf("global=%d workgroup=%d \n", global_offset[0],local_work_size[0]);
    errcode = clEnqueueNDRangeKernel(command_queue[1], kernel, work_dim, NULL, global_offset, local_work_size, 0, NULL, &(eventList[1]));
    if(errcode != CL_SUCCESS) printf("Error in cpu clEnqueueNDRangeKernel %d\n",errcode );
  }
  
//  printf("global[0]=%d sum=%d, remain_global_work_size[0]=%d, global_offset[0]=%d, local_work_size=%d\n",global_work_size[0],remain_global_work_size[0]+global_offset[0],remain_global_work_size[0],global_offset[0],local_work_size[0]);
  
  if(gpu_run) errcode = clFlush(command_queue[0]);
  if(cpu_run) errcode = clFlush(command_queue[1]);
  if(gpu_run) errcode = clWaitForEvents(1,&eventList[0]);
  if(cpu_run) errcode = clWaitForEvents(1,&eventList[1]);

  return errcode;
}


void computePhiMag_GPU(int numK,cl_mem phiR_d,cl_mem phiI_d,cl_mem phiMag_d,clPrmtr* clPrm)
{
  int phiMagBlocks = numK / KERNEL_PHI_MAG_THREADS_PER_BLOCK;
  if (numK % KERNEL_PHI_MAG_THREADS_PER_BLOCK)
    phiMagBlocks++;
  
  size_t DimPhiMagBlock = KERNEL_PHI_MAG_THREADS_PER_BLOCK;
  size_t DimPhiMagGrid = phiMagBlocks*KERNEL_PHI_MAG_THREADS_PER_BLOCK;

  cl_int clStatus;
  clStatus = clSetKernelArg(clPrm->clKernel,0,sizeof(cl_mem),&phiR_d);
  clStatus = clSetKernelArg(clPrm->clKernel,1,sizeof(cl_mem),&phiI_d);
  clStatus = clSetKernelArg(clPrm->clKernel,2,sizeof(cl_mem),&phiMag_d);
  clStatus = clSetKernelArg(clPrm->clKernel,3,sizeof(int),&numK);
  CHECK_ERROR("clSetKernelArg")

//  double  starttime = gettime();
  clStatus = clEnqueueNDRangeKernel(clPrm->clCommandQueue[0],clPrm->clKernel,1,NULL,&DimPhiMagGrid,&DimPhiMagBlock,0,NULL,NULL);
  CHECK_ERROR("clEnqueueNDRangeKernel")
   clFinish(clPrm->clCommandQueue[0]);
 //  double endtime = gettime();
  // printf("kernel1: %lf s\n", endtime -starttime);
   
}

static
unsigned long long int
readElapsedTime(cl_event internal)
{
  cl_int status;
  cl_ulong t_begin, t_end;
  status = clGetEventProfilingInfo(internal, CL_PROFILING_COMMAND_START,
    sizeof(cl_ulong), &t_begin, NULL);
  if (status != CL_SUCCESS) return 0;
  status = clGetEventProfilingInfo(internal, CL_PROFILING_COMMAND_END,
  sizeof(cl_ulong), &t_end, NULL);
  if (status != CL_SUCCESS) return 0;
  return (unsigned long long int)(t_end - t_begin);
}


void computeQ_GPU (int numK,int numX,
		   cl_mem x_d, cl_mem y_d, cl_mem z_d,
		   struct kValues* kVals,
		   cl_mem Qr_d, cl_mem Qi_d,
		   clPrmtr* clPrm)
{
  int QGrids = numK / KERNEL_Q_K_ELEMS_PER_GRID;
  if (numK % KERNEL_Q_K_ELEMS_PER_GRID)
    QGrids++;
  int QBlocks = numX / KERNEL_Q_THREADS_PER_BLOCK;
  if (numX % KERNEL_Q_THREADS_PER_BLOCK)
    QBlocks++;

  size_t DimQBlock = KERNEL_Q_THREADS_PER_BLOCK/NC;
  size_t DimQGrid = QBlocks*KERNEL_Q_THREADS_PER_BLOCK/NC;

  cl_int clStatus;
  cl_mem ck;
  ck = clCreateBuffer(clPrm->clContext,CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR,KERNEL_Q_K_ELEMS_PER_GRID*sizeof(struct kValues),NULL,&clStatus);

  int QGrid;
  for (QGrid = 0; QGrid < QGrids; QGrid++) {
    // Put the tile of K values into constant mem
    int QGridBase = QGrid * KERNEL_Q_K_ELEMS_PER_GRID;
    struct kValues* kValsTile = kVals + QGridBase;
    int numElems = MIN(KERNEL_Q_K_ELEMS_PER_GRID, numK - QGridBase);

    clStatus = clEnqueueWriteBuffer(clPrm->clCommandQueue[0],ck,CL_TRUE,0,numElems*sizeof(struct kValues),kValsTile,0,NULL,NULL);
    CHECK_ERROR("clEnqueueWriteBuffer")
    
    clStatus = clSetKernelArg(clPrm->clKernel,0,sizeof(int),&numK);
    clStatus = clSetKernelArg(clPrm->clKernel,1,sizeof(int),&QGridBase);
    clStatus = clSetKernelArg(clPrm->clKernel,2,sizeof(cl_mem),&x_d);
    clStatus = clSetKernelArg(clPrm->clKernel,3,sizeof(cl_mem),&y_d);
    clStatus = clSetKernelArg(clPrm->clKernel,4,sizeof(cl_mem),&z_d);
    clStatus = clSetKernelArg(clPrm->clKernel,5,sizeof(cl_mem),&Qr_d);
    clStatus = clSetKernelArg(clPrm->clKernel,6,sizeof(cl_mem),&Qi_d);
    clStatus = clSetKernelArg(clPrm->clKernel,7,sizeof(cl_mem),&ck);
    CHECK_ERROR("clSetKernelArg")

    //printf ("Grid: %d, Block: %d\n", DimQGrid, DimQBlock);

    #define TIMED_EXECUTION
    #ifdef TIMED_EXECUTION
    cl_event e;
//  double  starttime = gettime();
    clStatus = clEnqueueNDRangeKernel_fusion(clPrm->clCommandQueue,clPrm->clKernel,1,NULL,&DimQGrid,&DimQBlock,0,NULL,&e);
    //clStatus = clEnqueueNDRangeKernel(clPrm->clCommandQueue[0],clPrm->clKernel,1,NULL,&DimQGrid,&DimQBlock,0,NULL,&e);
    CHECK_ERROR("clEnqueueNDRangeKernel")
//   clFinish(clPrm->clCommandQueue[0]);//zhangfeng
 //  double endtime = gettime();
  // printf("kernel2 this one: %lf s\n", endtime -starttime);
   
//    clWaitForEvents(1, &e); 
 //exit(-1);   printf ("%llu\n", readElapsedTime(e));
    #else
//    starttime = gettime();
    clStatus = clEnqueueNDRangeKernel(clPrm->clCommandQueue[0],clPrm->clKernel,1,NULL,&DimQGrid,&DimQBlock,0,NULL,NULL);
    CHECK_ERROR("clEnqueueNDRangeKernel")
    clFinish(clPrm->clCommandQueue[0]);
 //   endtime = gettime();
  //  printf("kernel3: %lf s\n", endtime -starttime);
 
    #endif
  }
}

void createDataStructsCPU(int numK, int numX, float** phiMag,
	 float** Qr, float** Qi)
{
  *phiMag = (float* ) memalign(16, numK * sizeof(float));
  *Qr = (float*) memalign(16, numX * sizeof (float));
  *Qi = (float*) memalign(16, numX * sizeof (float));
}

