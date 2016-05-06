/**
 * gesummv.c: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define MAX_SOURCE_SIZE (0x100000)

/* Problem size */
#define N 8192

/* Thread block dimensions */
#define DIM_LOCAL_WORK_GROUP_X 64
//#define DIM_LOCAL_WORK_GROUP_X 256
#define DIM_LOCAL_WORK_GROUP_Y 1

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

char str_temp[1024];

DATA_TYPE ALPHA = 1;
DATA_TYPE BETA = 1;

cl_platform_id platform_id;
cl_device_id device_id[2];   
//cl_device_id device_id;   
cl_uint num_devices;
cl_uint num_platforms;
cl_int errcode;
cl_context clGPUContext;
cl_kernel clKernel1;
cl_kernel clKernel2;
cl_kernel clKernel3;
cl_command_queue clCommandQue[2];
//cl_command_queue clCommandQue;
cl_program clProgram;

cl_mem a_mem_obj;
cl_mem b_mem_obj;
cl_mem x_mem_obj;
cl_mem y_mem_obj;
cl_mem tmp_mem_obj;

FILE *fp;
char *source_str;
size_t source_size;

int cpu_offset = 50;


void compareResults(DATA_TYPE* y, DATA_TYPE* y_outputFromGpu)
{
	int i, fail;
	fail = 0;
	
	for (i=0; i<(N); i++) 
	{
		if (percentDiff(y[i], y_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD) 
		{
			fail++;
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);

}


void init(DATA_TYPE* A, DATA_TYPE* x)
{
  	int i, j;

 	for (i = 0; i < N; i++)
    	{
    		x[i] = ((DATA_TYPE) i) / N;
      	
		for (j = 0; j < N; j++) 
		{
			A[i*N + j] = ((DATA_TYPE) i*j) / N;
		}
    	}
}


void read_cl_file()
{
	// Load the kernel source code into the array source_str
	fp = fopen("gesummv.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );
}


void gesummv(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *tmp)
{
	int i,j;
	
	for (i = 0; i < N; i++)
    	{
		tmp[i] = 0;
		y[i] = 0;
		for (j = 0; j < N; j++)
		{
			tmp[i] = A[i*N + j] * x[j] + tmp[i];
			y[i] = B[i*N + j] * x[j] + y[i];
		}
		
		y[i] = ALPHA * tmp[i] + BETA * y[i];
	}
}


void cl_initialization_fusion()
{
	
	// Get platform and device information
	errcode = clGetPlatformIDs(1, &platform_id, &num_platforms);
	if(errcode == CL_SUCCESS) printf("number of platforms is %d\n",num_platforms);
	else printf("Error getting platform IDs\n");

	errcode = clGetPlatformInfo(platform_id,CL_PLATFORM_NAME, sizeof(str_temp), str_temp,NULL);
	if(errcode == CL_SUCCESS) printf("platform name is %s\n",str_temp);
	else printf("Error getting platform name\n");

	errcode = clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, sizeof(str_temp), str_temp,NULL);
	if(errcode == CL_SUCCESS) printf("platform version is %s\n",str_temp);
	else printf("Error getting platform version\n");

	errcode = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id[0], &num_devices);
	if(errcode == CL_SUCCESS) printf("number of GPU is %d\n", num_devices);
	errcode |= clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_CPU, 1, &device_id[1], &num_devices);
	if(errcode == CL_SUCCESS) printf("number of CPU is %d\n", num_devices);
	else printf("Error getting device IDs\n");

	errcode = clGetDeviceInfo(device_id[0],CL_DEVICE_NAME, sizeof(str_temp), str_temp,NULL);
	if(errcode == CL_SUCCESS) printf("GPU device name is %s\n",str_temp);
	else printf("GPU Error getting device name\n");
	errcode = clGetDeviceInfo(device_id[1],CL_DEVICE_NAME, sizeof(str_temp), str_temp,NULL);
	if(errcode == CL_SUCCESS) printf("CPU device name is %s\n",str_temp);
	else printf("CPU Error getting device name\n");
	
	// Create an OpenCL context
	clGPUContext = clCreateContext( NULL, 2, device_id, NULL, NULL, &errcode);
	//clGPUContext = clCreateContext( NULL, 1, &device_id, NULL, NULL, &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating context\n");
 
	//Create a command-queue
	clCommandQue[0] = clCreateCommandQueue(clGPUContext, device_id[0], 0, &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating command queue\n");
	clCommandQue[1] = clCreateCommandQueue(clGPUContext, device_id[1], 0, &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating command queue\n");
}

/*
void cl_initialization()
{	
	// Get platform and device information
	errcode = clGetPlatformIDs(1, &platform_id, &num_platforms);
	if(errcode == CL_SUCCESS) printf("number of platforms is %d\n",num_platforms);
	else printf("Error getting platform IDs\n");

	errcode = clGetPlatformInfo(platform_id,CL_PLATFORM_NAME, sizeof(str_temp), str_temp,NULL);
	if(errcode == CL_SUCCESS) printf("platform name is %s\n",str_temp);
	else printf("Error getting platform name\n");

	errcode = clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, sizeof(str_temp), str_temp,NULL);
	if(errcode == CL_SUCCESS) printf("platform version is %s\n",str_temp);
	else printf("Error getting platform version\n");

	errcode = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_devices);
	if(errcode == CL_SUCCESS) printf("number of devices is %d\n", num_devices);
	else printf("Error getting device IDs\n");

	errcode = clGetDeviceInfo(device_id,CL_DEVICE_NAME, sizeof(str_temp), str_temp,NULL);
	if(errcode == CL_SUCCESS) printf("device name is %s\n",str_temp);
	else printf("Error getting device name\n");
	
	// Create an OpenCL context
	clGPUContext = clCreateContext( NULL, 1, &device_id, NULL, NULL, &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating context\n");
 
	//Create a command-queue
	clCommandQue = clCreateCommandQueue(clGPUContext, device_id, 0, &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating command queue\n");
}
*/


void cl_mem_init(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *tmp)
{
	a_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, sizeof(DATA_TYPE) * N * N, NULL, &errcode);
	b_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, sizeof(DATA_TYPE) * N * N, NULL, &errcode);
	x_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, sizeof(DATA_TYPE) * N, NULL, &errcode);
	y_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, sizeof(DATA_TYPE) * N, NULL, &errcode);
	tmp_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, sizeof(DATA_TYPE) * N, NULL, &errcode);
	
	if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");

	errcode = clEnqueueWriteBuffer(clCommandQue[0], a_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N * N, A, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue[0], b_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N * N, B, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue[0], x_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N, x, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue[0], y_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N, y, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue[0], tmp_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N, tmp, 0, NULL, NULL);
	if(errcode != CL_SUCCESS)printf("Error in writing buffers\n");
}


void cl_load_prog()
{
	// Create a program from the kernel source
	clProgram = clCreateProgramWithSource(clGPUContext, 1, (const char **)&source_str, (const size_t *)&source_size, &errcode);

	if(errcode != CL_SUCCESS) printf("Error in creating program\n");

	// Build the program
	errcode = clBuildProgram(clProgram, 0, 0, NULL, NULL, NULL);
	//errcode = clBuildProgram(clProgram, 1, &device_id, NULL, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in building program\n");
		
	// Create the OpenCL kernel
	clKernel1 = clCreateKernel(clProgram, "gesummv_kernel", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel1\n");
	clFinish(clCommandQue[0]);
}

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
  remain_global_work_size[0] = global_work_size[0]-global_offset[0];
  if(remain_global_work_size[0] == 0)
    gpu_run = 0;
  global_offset_start[0]=global_offset[0];
  global_offset_start[1]=0;
  if(gpu_run){
    errcode = clEnqueueNDRangeKernel(command_queue[0], kernel, work_dim, global_offset_start, remain_global_work_size, local_work_size, 0, NULL, &(eventList[0]));
    if(errcode != CL_SUCCESS) printf("Error in gpu clEnqueueNDRangeKernel\n");
  }
//  clFinish(command_queue[0]);
  if(cpu_run){
    errcode = clEnqueueNDRangeKernel(command_queue[1], kernel, work_dim, NULL, global_offset, local_work_size, 0, NULL, &(eventList[1]));
    if(errcode != CL_SUCCESS) printf("Error in cpu clEnqueueNDRangeKernel\n");
  }
  
  if(gpu_run) errcode = clFlush(command_queue[0]);
  if(cpu_run) errcode = clFlush(command_queue[1]);
  if(gpu_run) errcode = clWaitForEvents(1,&eventList[0]);
  if(cpu_run) errcode = clWaitForEvents(1,&eventList[1]);

  return errcode;
}



void cl_launch_kernel()
{
	double t_start, t_end;

	int n = N;

	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
	localWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;
	globalWorkSize[0] = (size_t)ceil(((float)N) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;
	globalWorkSize[1] = (size_t)ceil(((float)N) / ((float)DIM_LOCAL_WORK_GROUP_Y)) * DIM_LOCAL_WORK_GROUP_Y;

	t_start = rtclock();
	
	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel1, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 1, sizeof(cl_mem), (void *)&b_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 2, sizeof(cl_mem), (void *)&x_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 3, sizeof(cl_mem), (void *)&y_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 4, sizeof(cl_mem), (void *)&tmp_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 5, sizeof(DATA_TYPE), (void *)&ALPHA);
	errcode |= clSetKernelArg(clKernel1, 6, sizeof(DATA_TYPE), (void *)&BETA);
	errcode |= clSetKernelArg(clKernel1, 7, sizeof(int), (void *)&n);

	if(errcode != CL_SUCCESS) printf("Error in seting arguments1\n");

	// Execute the OpenCL kernel
	errcode = clEnqueueNDRangeKernel_fusion(clCommandQue, clKernel1, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	//errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel1, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel1\n");
//	clFinish(clCommandQue);

	t_end = rtclock();
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
	fprintf(stdout, "CAUTION:CPU offset %d %% GPU Runtime: %0.6lf s\n",cpu_offset, t_end - t_start);
}


void cl_clean_up()
{
	// Clean up
	errcode = clFlush(clCommandQue[0]);
	errcode = clFinish(clCommandQue[1]);
	errcode = clFlush(clCommandQue[0]);
	errcode = clFinish(clCommandQue[1]);

	errcode = clReleaseKernel(clKernel1);
	errcode = clReleaseProgram(clProgram);
	errcode = clReleaseMemObject(a_mem_obj);
	errcode = clReleaseMemObject(b_mem_obj);
	errcode = clReleaseMemObject(x_mem_obj);
	errcode = clReleaseCommandQueue(clCommandQue[0]);
	errcode = clReleaseCommandQueue(clCommandQue[1]);
	errcode = clReleaseContext(clGPUContext);
	if(errcode != CL_SUCCESS) printf("Error in cleanup\n");
}


int main(int argc, char* argv[]) 
//int main(void) 
{
	double t_start, t_end;

	DATA_TYPE* A;
	DATA_TYPE* B;  
	DATA_TYPE* x;  
	DATA_TYPE* y;
	DATA_TYPE* y_outputFromGpu;
	DATA_TYPE* tmp;
	if(argc==2){
          printf("arg 1 = %s\narg 2 = %s\n", argv[0], argv[1]);
          cpu_offset = atoi(argv[1]);
        }


	A = (DATA_TYPE*)malloc(N*N*sizeof(DATA_TYPE));
	B = (DATA_TYPE*)malloc(N*N*sizeof(DATA_TYPE));
	x = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE)); 
	y = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
	y_outputFromGpu = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
	tmp = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));

	init(A, x);
	read_cl_file();
	cl_initialization_fusion();
	//cl_initialization();
	cl_mem_init(A, B, x, y, tmp);
	cl_load_prog();

	cl_launch_kernel();

	errcode = clEnqueueReadBuffer(clCommandQue[0], y_mem_obj, CL_TRUE, 0, N*sizeof(DATA_TYPE), y_outputFromGpu, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in reading GPU mem\n");

	t_start = rtclock();
	gesummv(A, B, x, y, tmp);
	t_end = rtclock(); 
	fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);   
	compareResults(y, y_outputFromGpu);
	cl_clean_up();
	
	free(A);
	free(B);  
	free(x);  
	free(y);
	free(y_outputFromGpu);
	free(tmp);

	return 0;
}

