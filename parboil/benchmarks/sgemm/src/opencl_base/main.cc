/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/* 
 * Main entry of dense matrix-matrix multiplication kernel
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <malloc.h>
#include <vector>
#include <iostream>
#include <CL/cl.h>
#include <parboil.h>
#include<sys/time.h>
double gettime() {
    struct timeval t;
      gettimeofday(&t,NULL);
        return t.tv_sec+t.tv_usec*1e-6;
}
// I/O routines
extern bool readColMajorMatrixFile(const char *fn, int &nr_row, int &nr_col, std::vector<float>&v);
extern bool writeColMajorMatrixFile(const char *fn, int, int, std::vector<float>&);
extern char* readFile(const char*);

// Parameters of tile sizes
#define TILE_SZ 16

#define CHECK_ERROR(errorMessage)           \
  if(clStatus != CL_SUCCESS)                \
  {                                         \
     std::cout<<errorMessage<<" Error!\n";  \
     std::cout<<"Line: "<<__LINE__<<"\n";   \
     exit(1);                               \
  }
int cpu_offset ;
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


void basicSgemm( char transa, char transb, int m, int n, int k, float alpha, cl_mem A, int lda, cl_mem B, int ldb, float beta, cl_mem C, int ldc, cl_kernel clKernel, cl_command_queue *clCommandQueue )
{
  if ((transa != 'N') && (transa != 'n')) {
    std::cerr << "unsupported value of 'transa' in regtileSgemm()" << std::endl;
    return;
  }
  
  if ((transb != 'T') && (transb != 't')) {
    std::cerr << "unsupported value of 'transb' in regtileSgemm()" << std::endl;
    return;
  }
  
  // In this code we assume the matrix sizes are multiple of tile size
  if ((m%TILE_SZ) || (n%TILE_SZ)) {
    std::cerr << "unsupported size of matrix. m should be multiple of " << TILE_SZ
      << "; n should be multiple of " << TILE_SZ << std::endl;
  }

  size_t db[2] = {TILE_SZ,TILE_SZ};
  size_t dg[2] = {m/TILE_SZ*db[0],n/TILE_SZ*db[1]};

  cl_int clStatus;
 
  clStatus = clSetKernelArg(clKernel,0,sizeof(cl_mem),(void*)&A);
  clStatus = clSetKernelArg(clKernel,1,sizeof(int),(void*)&lda);
  clStatus = clSetKernelArg(clKernel,2,sizeof(cl_mem),(void*)&B);
  clStatus = clSetKernelArg(clKernel,3,sizeof(int),(void*)&ldb);
  clStatus = clSetKernelArg(clKernel,4,sizeof(cl_mem),(void*)&C);
  clStatus = clSetKernelArg(clKernel,5,sizeof(int),(void*)&ldc);
  clStatus = clSetKernelArg(clKernel,6,sizeof(int),(void*)&k);
  clStatus = clSetKernelArg(clKernel,7,sizeof(float),(void*)&alpha);
  clStatus = clSetKernelArg(clKernel,8,sizeof(float),(void*)&beta);
  CHECK_ERROR("clSetKernelArg")

  clStatus = clEnqueueNDRangeKernel_fusion(clCommandQueue,clKernel,2,NULL,dg,db,0,NULL,NULL);
  //clStatus = clEnqueueNDRangeKernel(clCommandQueue[0],clKernel,2,NULL,dg,db,0,NULL,NULL);
  CHECK_ERROR("clEnqueueNDRangeKernel")

  clStatus = clFinish(clCommandQueue[0]); 
  CHECK_ERROR("clFinish")
}

main (int argc, char *argv[]) {

  struct pb_Parameters *params;
  struct pb_TimerSet timers;

  size_t A_sz, B_sz, C_sz;
  int matArow, matAcol;
  int matBrow, matBcol;
  std::vector<float> matA, matBT;

  pb_InitializeTimerSet(&timers);

  /* Read command line. Expect 3 inputs: A, B and B^T 
     in column-major layout*/
  params = pb_ReadParameters(&argc, argv);

  printf("percentage: %d\n", params->percentage);//zf zhangfeng
  cpu_offset=params->percentage;
 

  if ((params->inpFiles[0] == NULL) 
      || (params->inpFiles[1] == NULL)
      || (params->inpFiles[2] == NULL)
      || (params->inpFiles[3] != NULL))
    {
      fprintf(stderr, "Expecting three input filenames\n");
      exit(-1);
    }

  cl_int clStatus;
  cl_platform_id clPlatform;
  clStatus = clGetPlatformIDs(1,&clPlatform,NULL);
  CHECK_ERROR("clGetPlatformIDs")

  ////////////////////////////////////////zhangfeng zf////////////////////////////
/*
  cl_context_properties clCps[3] = {CL_CONTEXT_PLATFORM,(cl_context_properties)clPlatform,0};
  cl_context clContext = clCreateContextFromType(clCps,CL_DEVICE_TYPE_GPU,NULL,NULL,&clStatus);
  CHECK_ERROR("clCreateContextFromType")
   
  cl_device_id clDevice;
  clStatus = clGetDeviceIDs(clPlatform,CL_DEVICE_TYPE_GPU,1,&clDevice,NULL);
  CHECK_ERROR("clGetDeviceIDs")

  cl_command_queue clCommandQueue = clCreateCommandQueue(clContext,clDevice,CL_QUEUE_PROFILING_ENABLE,&clStatus);
  CHECK_ERROR("clCreateCommandQueue")
*/
  ////////////////////////////////////////zhangfeng zf////////////////////////////

        cl_int errcode;
        cl_uint num_devices;
  cl_device_id clDevice[2];
        char str_temp[1024];
        cl_command_queue clCommandQueue[2];
        errcode = clGetDeviceIDs( clPlatform, CL_DEVICE_TYPE_GPU, 1, &clDevice[0], &num_devices);
        if(errcode == CL_SUCCESS) printf("number of GPU is %d\n", num_devices);
        errcode |= clGetDeviceIDs( clPlatform, CL_DEVICE_TYPE_CPU, 1, &clDevice[1], &num_devices);
        if(errcode == CL_SUCCESS) printf("number of CPU is %d\n", num_devices);
        else printf("Error getting device IDs\n");

        errcode = clGetDeviceInfo(clDevice[0],CL_DEVICE_NAME, sizeof(str_temp), str_temp,NULL);
        if(errcode == CL_SUCCESS) printf("GPU device name is %s\n",str_temp);
        else printf("GPU Error getting device name\n");
        errcode = clGetDeviceInfo(clDevice[1],CL_DEVICE_NAME, sizeof(str_temp), str_temp,NULL);
        if(errcode == CL_SUCCESS) printf("CPU device name is %s\n",str_temp);
        else printf("CPU Error getting device name\n");
     
        // Create an OpenCL context
        cl_context clContext = clCreateContext( NULL, 2, clDevice, NULL, NULL, &errcode);
        if(errcode != CL_SUCCESS) printf("Error in creating context\n");
 
        //Create a command-queue
        clCommandQueue[0] = clCreateCommandQueue(clContext, clDevice[0], CL_QUEUE_PROFILING_ENABLE, &errcode);
        if(errcode != CL_SUCCESS) printf("Error in creating command queue\n");
        clCommandQueue[1] = clCreateCommandQueue(clContext, clDevice[1], CL_QUEUE_PROFILING_ENABLE, &errcode);
        if(errcode != CL_SUCCESS) printf("Error in creating command queue\n");


  ////////////////////////////////////////zhangfeng zf////////////////////////////
  


  pb_SetOpenCL(&clContext, &clCommandQueue[0]);
  //pb_SetOpenCL(&clContext, &clCommandQueue);

  const char* clSource[] = {readFile("src/opencl_base/kernel.cl")};
  cl_program clProgram = clCreateProgramWithSource(clContext,1,clSource,NULL,&clStatus);
  CHECK_ERROR("clCreateProgramWithSource")

  char clOptions[50];
  sprintf(clOptions,"");

  clStatus = clBuildProgram(clProgram,0,0,clOptions,NULL,NULL);
  //clStatus = clBuildProgram(clProgram,1,&clDevice,clOptions,NULL,NULL);
  CHECK_ERROR("clBuildProgram")

  cl_kernel clKernel = clCreateKernel(clProgram,"mysgemmNT",&clStatus);
  CHECK_ERROR("clCreateKernel")

  /* Read in data */
  pb_SwitchToTimer(&timers, pb_TimerID_IO);

  // load A
  readColMajorMatrixFile(params->inpFiles[0],
      matArow, matAcol, matA);
  // copy A to device memory
  A_sz = matArow*matAcol*sizeof(float);

  // load B^T
  readColMajorMatrixFile(params->inpFiles[2],
      matBcol, matBrow, matBT);

  pb_SwitchToTimer( &timers, pb_TimerID_COMPUTE );
  B_sz = matBrow*matBcol*sizeof(float);

  // allocate space for C
  C_sz = matArow*matBcol*sizeof(float);

  // OpenCL memory allocation
  std::vector<float> matC(matArow*matBcol);
  cl_mem dA = clCreateBuffer(clContext,CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR,A_sz,NULL,&clStatus);
  CHECK_ERROR("clCreateBuffer")
  cl_mem dB = clCreateBuffer(clContext,CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR,B_sz,NULL,&clStatus);
  CHECK_ERROR("clCreateBuffer")
  cl_mem dC = clCreateBuffer(clContext,CL_MEM_WRITE_ONLY|CL_MEM_ALLOC_HOST_PTR,C_sz,NULL,&clStatus);
  CHECK_ERROR("clCreateBuffer")

  // Copy A and B^T into device memory
  pb_SwitchToTimer( &timers, pb_TimerID_COPY );
  clStatus = clEnqueueWriteBuffer(clCommandQueue[0],dA,CL_FALSE,0,A_sz,&matA.front(),0,NULL,NULL);
  CHECK_ERROR("clEnqueueWriteBuffer")
  clStatus = clEnqueueWriteBuffer(clCommandQueue[0],dB,CL_FALSE,0,B_sz,&matBT.front(),0,NULL,NULL);
  CHECK_ERROR("clEnqueueWriteBuffer")

  for(int i=0;i<matC.size();i++)
	matC[i] = 0.0f;

  clStatus = clEnqueueWriteBuffer(clCommandQueue[0],dC,CL_TRUE,0,C_sz,&matC.front(),0,NULL,NULL);
  CHECK_ERROR("clEnqueueWriteBuffer")

  pb_SwitchToTimer( &timers, pb_TimerID_KERNEL );

  // Use standard sgemm interface
    double starttime = gettime();
  basicSgemm('N', 'T', matArow, matBcol, matAcol, 1.0f, \
      dA, matArow, dB, matBcol, 0.0f, dC, matArow, clKernel, clCommandQueue);
   double  endtime = gettime();
   printf("CAUTION: percent %d time %lf s \n",cpu_offset, endtime -starttime);
 
  if (params->outFile) {
    pb_SwitchToTimer( &timers, pb_TimerID_COPY );
    clEnqueueReadBuffer(clCommandQueue[0],dC,CL_TRUE,0,C_sz,&matC.front(),0,NULL,NULL);
   
    /* Write C to file */
    pb_SwitchToTimer(&timers, pb_TimerID_IO);
    writeColMajorMatrixFile(params->outFile,
	matArow, matBcol, matC); 
  }

  pb_SwitchToTimer(&timers, pb_TimerID_NONE);

  double GPUtime = pb_GetElapsedTime(&(timers.timers[pb_TimerID_KERNEL]));
  std::cout<< "GFLOPs = " << 2.* matArow * matBcol * matAcol/GPUtime/1e9 << std::endl;
  pb_PrintTimerSet(&timers);
  pb_FreeParameters(params);

  free((void*)clSource[0]);

  clStatus = clReleaseKernel(clKernel);
  clStatus = clReleaseProgram(clProgram);
  clStatus = clReleaseMemObject(dA);
  clStatus = clReleaseMemObject(dB);
  clStatus = clReleaseMemObject(dC);
  clStatus = clReleaseCommandQueue(clCommandQueue[0]);
  clStatus = clReleaseContext(clContext); 
  
  return 0;
}
