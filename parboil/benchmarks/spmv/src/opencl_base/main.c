/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <stdio.h>
#include <stdlib.h>
#include <parboil.h>



#include "file.h"
#include "gpu_info.h"
#include "ocl.h"
#include "convert_dataset.h"
double gettime() {
    struct timeval t;
      gettimeofday(&t,NULL);
        return t.tv_sec+t.tv_usec*1e-6;
}
int cpu_offset;

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


static int generate_vector(float *x_vector, int dim) 
{	
	srand(54321);
	int i;
	for(i=0;i<dim;i++)
	{
		x_vector[i] = (rand() / (float) RAND_MAX);
	}
	return 0;
}

int main(int argc, char** argv) {
	struct pb_TimerSet timers;
	struct pb_Parameters *parameters;
	
	printf("CUDA accelerated sparse matrix vector multiplication****\n");
	printf("Original version by Li-Wen Chang <lchang20@illinois.edu> and Shengzhao Wu<wu14@illinois.edu>\n");
	printf("This version maintained by Chris Rodrigues  ***********\n");
	parameters = pb_ReadParameters(&argc, argv);

  printf("percentage: %d\n", parameters->percentage);//zf zhangfeng
  cpu_offset=parameters->percentage;
 
	if ((parameters->inpFiles[0] == NULL) || (parameters->inpFiles[1] == NULL))
    	{
      		fprintf(stderr, "Expecting one input filename\n");
      		exit(-1);
    	}

	pb_InitializeTimerSet(&timers);
	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	
	//parameters declaration
	cl_int clStatus;
	cl_platform_id clPlatform;
	clStatus = clGetPlatformIDs(1,&clPlatform,NULL);
	CHECK_ERROR("clGetPlatformIDs")

  ////////////////////////////////////////zhangfeng zf////////////////////////////
/*
	cl_context_properties clCps[3] = {CL_CONTEXT_PLATFORM,(cl_context_properties)clPlatform,0};
	
	cl_device_id clDevice;
	clStatus = clGetDeviceIDs(clPlatform,CL_DEVICE_TYPE_GPU,1,&clDevice,NULL);
	CHECK_ERROR("clGetDeviceIDs")

	cl_context clContext = clCreateContextFromType(clCps,CL_DEVICE_TYPE_GPU,NULL,NULL,&clStatus);
	CHECK_ERROR("clCreateContextFromType")

	cl_command_queue clCommandQueue = clCreateCommandQueue(clContext,clDevice,CL_QUEUE_PROFILING_ENABLE,&clStatus);
	CHECK_ERROR("clCreateCommandQueue")
        */
  ////////////////////////////////////////zhangfeng zf////////////////////////////

        cl_int errcode;
        cl_uint num_devices;
        char str_temp[1024];
	cl_device_id clDevice[2];
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
  pb_SetOpenCL(&clContext, &clCommandQueue[0]);

  ////////////////////////////////////////zhangfeng zf////////////////////////////




  //	pb_SetOpenCL(&clContext, &clCommandQueue);
	
	const char* clSource[] = {readFile("src/opencl_base/kernel.cl")};
	cl_program clProgram = clCreateProgramWithSource(clContext,1,clSource,NULL,&clStatus);
	CHECK_ERROR("clCreateProgramWithSource")

	char clOptions[50];
	sprintf(clOptions,"");
	clStatus = clBuildProgram(clProgram,0,0,clOptions,NULL,NULL);
	//clStatus = clBuildProgram(clProgram,1,&clDevice,clOptions,NULL,NULL);
	CHECK_ERROR("clBuildProgram")

	cl_kernel clKernel = clCreateKernel(clProgram,"spmv_jds_naive",&clStatus);
	CHECK_ERROR("clCreateKernel")

	int len;
	int depth;
	int dim;
	int pad=32;
	int nzcnt_len;
	
	//host memory allocation
	//matrix
	float *h_data;
	int *h_indices;
	int *h_ptr;
	int *h_perm;
	int *h_nzcnt;
	//vector
	float *h_Ax_vector;
	float *h_x_vector;
	
	//device memory allocation
	//matrix
	cl_mem d_data;
	cl_mem d_indices;
	cl_mem d_ptr;
	cl_mem d_perm;
	cl_mem d_nzcnt;

	//vector
	cl_mem d_Ax_vector;
	cl_mem d_x_vector;
	
	cl_mem jds_ptr_int;
	cl_mem sh_zcnt_int;

    	//load matrix from files
	pb_SwitchToTimer(&timers, pb_TimerID_IO);
	//inputData(parameters->inpFiles[0], &len, &depth, &dim,&nzcnt_len,&pad,
	//    &h_data, &h_indices, &h_ptr,
	//    &h_perm, &h_nzcnt);
	int col_count;
	coo_to_jds(
		parameters->inpFiles[0], // bcsstk32.mtx, fidapm05.mtx, jgl009.mtx
		1, // row padding
		pad, // warp size
		1, // pack size
		1, // is mirrored?
		0, // binary matrix
		1, // debug level [0:2]
		&h_data, &h_ptr, &h_nzcnt, &h_indices, &h_perm,
		&col_count, &dim, &len, &nzcnt_len, &depth
	);
	
//	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	h_Ax_vector=(float*)malloc(sizeof(float)*dim);	
	h_x_vector=(float*)malloc(sizeof(float)*dim);	
	
  input_vec( parameters->inpFiles[1],h_x_vector,dim);

	 pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
    	OpenCLDeviceProp clDeviceProp;
//	clStatus = clGetDeviceInfo(clDevice,CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV,sizeof(cl_uint),&(clDeviceProp.major),NULL);
	//CHECK_ERROR("clGetDeviceInfo")
//	clStatus = clGetDeviceInfo(clDevice,CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV,sizeof(cl_uint),&(clDeviceProp.minor),NULL);
  //      CHECK_ERROR("clGetDeviceInfo")
	clStatus = clGetDeviceInfo(clDevice[0],CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cl_uint),&(clDeviceProp.multiProcessorCount),NULL);
        CHECK_ERROR("clGetDeviceInfo")
	
	pb_SwitchToTimer(&timers, pb_TimerID_COPY);
	//memory allocation
	d_data = clCreateBuffer(clContext,CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR,len*sizeof(float),NULL,&clStatus);
	CHECK_ERROR("clCreateBuffer")
	d_indices = clCreateBuffer(clContext,CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR,len*sizeof(int),NULL,&clStatus);
	CHECK_ERROR("clCreateBuffer")
	d_perm = clCreateBuffer(clContext,CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR,dim*sizeof(int),NULL,&clStatus);
	CHECK_ERROR("clCreateBuffer")
	d_x_vector = clCreateBuffer(clContext,CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR,dim*sizeof(float),NULL,&clStatus);
	CHECK_ERROR("clCreateBuffer")
	d_Ax_vector = clCreateBuffer(clContext,CL_MEM_WRITE_ONLY|CL_MEM_ALLOC_HOST_PTR,dim*sizeof(float),NULL,&clStatus);
	CHECK_ERROR("clCreateBuffer")

	jds_ptr_int = clCreateBuffer(clContext,CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR,5000*sizeof(int),NULL,&clStatus);
	CHECK_ERROR("clCreateBuffer")
	sh_zcnt_int = clCreateBuffer(clContext,CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR,5000*sizeof(int),NULL,&clStatus);
	CHECK_ERROR("clCreateBuffer")

	clMemSet(clCommandQueue[0],d_Ax_vector,0,dim*sizeof(float));
	
	//memory copy
	clStatus = clEnqueueWriteBuffer(clCommandQueue[0],d_data,CL_FALSE,0,len*sizeof(float),h_data,0,NULL,NULL);
	CHECK_ERROR("clEnqueueWriteBuffer")
	clStatus = clEnqueueWriteBuffer(clCommandQueue[0],d_indices,CL_FALSE,0,len*sizeof(int),h_indices,0,NULL,NULL);
	CHECK_ERROR("clEnqueueWriteBuffer")
	clStatus = clEnqueueWriteBuffer(clCommandQueue[0],d_perm,CL_FALSE,0,dim*sizeof(int),h_perm,0,NULL,NULL);
	CHECK_ERROR("clEnqueueWriteBuffer")
	clStatus = clEnqueueWriteBuffer(clCommandQueue[0],d_x_vector,CL_FALSE,0,dim*sizeof(int),h_x_vector,0,NULL,NULL);
	CHECK_ERROR("clEnqueueWriteBuffer")

	clStatus = clEnqueueWriteBuffer(clCommandQueue[0],jds_ptr_int,CL_FALSE,0,depth*sizeof(int),h_ptr,0,NULL,NULL);
	CHECK_ERROR("clEnqueueWriteBuffer")
	clStatus = clEnqueueWriteBuffer(clCommandQueue[0],sh_zcnt_int,CL_TRUE,0,nzcnt_len*sizeof(int),h_nzcnt,0,NULL,NULL);
	CHECK_ERROR("clEnqueueWriteBuffer")
	
	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	
	size_t grid;
	size_t block;

	compute_active_thread(&block,&grid,nzcnt_len,pad,clDeviceProp.major,clDeviceProp.minor,clDeviceProp.multiProcessorCount);
//  printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!grid is %d and block is %d=\n",grid,block);
//  printf("!!! dim is %d\n",dim);

	clStatus = clSetKernelArg(clKernel,0,sizeof(cl_mem),&d_Ax_vector);
	CHECK_ERROR("clSetKernelArg")
	clStatus = clSetKernelArg(clKernel,1,sizeof(cl_mem),&d_data);
	CHECK_ERROR("clSetKernelArg")
	clStatus = clSetKernelArg(clKernel,2,sizeof(cl_mem),&d_indices);
	CHECK_ERROR("clSetKernelArg")
	clStatus = clSetKernelArg(clKernel,3,sizeof(cl_mem),&d_perm);
	CHECK_ERROR("clSetKernelArg")
	clStatus = clSetKernelArg(clKernel,4,sizeof(cl_mem),&d_x_vector);
	CHECK_ERROR("clSetKernelArg")
	clStatus = clSetKernelArg(clKernel,5,sizeof(int),&dim);
	CHECK_ERROR("clSetKernelArg")

	clStatus = clSetKernelArg(clKernel,6,sizeof(cl_mem),&jds_ptr_int);
	CHECK_ERROR("clSetKernelArg")
	clStatus = clSetKernelArg(clKernel,7,sizeof(cl_mem),&sh_zcnt_int);
        CHECK_ERROR("clSetKernelArg")

	//main execution
	pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);

    double starttime = gettime();
	int i;
	for (i=0; i<50; i++)
	{
		clStatus = clEnqueueNDRangeKernel_fusion(clCommandQueue,clKernel,1,NULL,&grid,&block,0,NULL,NULL);
		//clStatus = clEnqueueNDRangeKernel(clCommandQueue[0],clKernel,1,NULL,&grid,&block,0,NULL,NULL);
		CHECK_ERROR("clEnqueueNDRangeKernel")
	}

   double  endtime = gettime();
   printf("CAUTION: percent %d time %lf s \n",cpu_offset, endtime -starttime);
	clStatus = clFinish(clCommandQueue[0]);
	CHECK_ERROR("clFinish")
	
	pb_SwitchToTimer(&timers, pb_TimerID_COPY);
	//HtoD memory copy
	clStatus = clEnqueueReadBuffer(clCommandQueue[0],d_Ax_vector,CL_TRUE,0,dim*sizeof(float),h_Ax_vector,0,NULL,NULL);
	CHECK_ERROR("clEnqueueReadBuffer")	

	clStatus = clReleaseKernel(clKernel);
	clStatus = clReleaseProgram(clProgram);

	clStatus = clReleaseMemObject(d_data);
	clStatus = clReleaseMemObject(d_indices);
        clStatus = clReleaseMemObject(d_perm);
	clStatus = clReleaseMemObject(d_nzcnt);
        clStatus = clReleaseMemObject(d_x_vector);
	clStatus = clReleaseMemObject(d_Ax_vector);
	CHECK_ERROR("clReleaseMemObject")

	clStatus = clReleaseCommandQueue(clCommandQueue[0]);
	clStatus = clReleaseCommandQueue(clCommandQueue[1]);
	clStatus = clReleaseContext(clContext);	
	
	if (parameters->outFile) {
		pb_SwitchToTimer(&timers, pb_TimerID_IO);
		outputData(parameters->outFile,h_Ax_vector,dim);
	}

	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	
	free((void*)clSource[0]);
	
	free (h_data);
	free (h_indices);
	free (h_ptr);
	free (h_perm);
	free (h_nzcnt);
	free (h_Ax_vector);
	free (h_x_vector);
	pb_SwitchToTimer(&timers, pb_TimerID_NONE);

	pb_PrintTimerSet(&timers);
	pb_FreeParameters(parameters);

	return 0;
}
