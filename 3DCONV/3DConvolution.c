/**
 * 3DConvolution.c: This file is part of the PolyBench/GPU 1.0 test suite.
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
#define PERCENT_DIFF_ERROR_THRESHOLD 1.05

#define MAX_SOURCE_SIZE (0x100000)

/* Problem size */
#define NI 256
#define NJ 256
#define NK 256

/* Thread block dimensions */
#define DIM_LOCAL_WORK_GROUP_X 8
//#define DIM_LOCAL_WORK_GROUP_X 32
#define DIM_LOCAL_WORK_GROUP_Y 8

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;


char str_temp[1024];


cl_platform_id platform_id;
cl_device_id device_id[2];   
//cl_device_id device_id;   
cl_uint num_devices;
cl_uint num_platforms;
cl_int errcode;
cl_context clGPUContext;
cl_kernel clKernel;
cl_command_queue clCommandQue[2];
//cl_command_queue clCommandQue;
cl_program clProgram;
cl_mem a_mem_obj;
cl_mem b_mem_obj;
FILE *fp;
char *source_str;
size_t source_size;

int cpu_offset = 50;


void read_cl_file()
{
	// Load the kernel source code into the array source_str
	fp = fopen("3DConvolution.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );
}


void init(DATA_TYPE* A)
{
	int i, j, k;

	for (i = 0; i < NI; ++i)
    	{
		for (j = 0; j < NJ; ++j)
		{
			for (k = 0; k < NK; ++k)
			{
				A[i*(NK * NJ) + j*NK + k] = i % 12 + 2 * (j % 7) + 3 * (k % 13);
			}
		}
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


void cl_mem_init(DATA_TYPE* A, DATA_TYPE* B)
{
	a_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR, sizeof(DATA_TYPE) * NI * NJ * NK, NULL, &errcode);
	b_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, sizeof(DATA_TYPE) * NI * NJ * NK, NULL, &errcode);
	
	if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");

	errcode = clEnqueueWriteBuffer(clCommandQue[0], a_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NI * NJ * NK, A, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue[0], b_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NI * NJ * NK, B, 0, NULL, NULL);
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
	clKernel = clCreateKernel(clProgram, "Convolution3D_kernel", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel\n");
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

	int ni = NI;
	int nj = NJ;
	int nk = NK;

	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
	localWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;
	globalWorkSize[0] = (size_t)ceil(((float)NK) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;
	globalWorkSize[1] = (size_t)ceil(((float)NJ) / ((float)DIM_LOCAL_WORK_GROUP_Y)) * DIM_LOCAL_WORK_GROUP_Y;

	t_start = rtclock();
	
	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	errcode |= clSetKernelArg(clKernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
	errcode |= clSetKernelArg(clKernel, 2, sizeof(int), &ni);
	errcode |= clSetKernelArg(clKernel, 3, sizeof(int), &nj);
	errcode |= clSetKernelArg(clKernel, 4, sizeof(int), &nk);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments\n");
	
	int i;
	for (i = 1; i < NI - 1; ++i) // 0
	{
		// set the current value of 'i' for the argument in the kernel
		errcode |= clSetKernelArg(clKernel, 5, sizeof(int), &i);

		// Execute the OpenCL kernel
		errcode = clEnqueueNDRangeKernel_fusion(clCommandQue, clKernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
		//errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	}

	if(errcode != CL_SUCCESS) printf("Error in launching kernel\n");
//	clFinish(clCommandQue);

	t_end = rtclock();
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
	fprintf(stdout, "CAUTION:CPU offset %d %% GPU Runtime: %0.6lf s\n",cpu_offset, t_end - t_start);
}


void cl_clean_up()
{
	// Clean up
	errcode = clFlush(clCommandQue[0]);
	errcode = clFlush(clCommandQue[1]);
	errcode = clFinish(clCommandQue[0]);
	errcode = clFinish(clCommandQue[1]);
	errcode = clReleaseKernel(clKernel);
	errcode = clReleaseProgram(clProgram);
	errcode = clReleaseMemObject(a_mem_obj);
	errcode = clReleaseMemObject(b_mem_obj);
	errcode = clReleaseCommandQueue(clCommandQue[0]);
	errcode = clReleaseCommandQueue(clCommandQue[1]);
	errcode = clReleaseContext(clGPUContext);
	if(errcode != CL_SUCCESS) printf("Error in cleanup\n");
}


void compareResults(DATA_TYPE* B, DATA_TYPE* B_outputFromGpu)
{
	int i, j, k, fail;
	fail = 0;
	
	// Compare result from cpu and gpu...
	for (i = 1; i < NI - 1; ++i) // 0
	{
		for (j = 1; j < NJ - 1; ++j) // 1
		{
			for (k = 1; k < NK - 1; ++k) // 2
			{
				if (percentDiff(B[i*(NK * NJ) + j*NK + k], B_outputFromGpu[i*(NK * NJ) + j*NK + k]) > PERCENT_DIFF_ERROR_THRESHOLD)
				{
					fail++;
				}
			}	
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);

}


void conv3D(DATA_TYPE* A, DATA_TYPE* B)
{
	int i, j, k;
	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +2;  c21 = +5;  c31 = -8;
	c12 = -3;  c22 = +6;  c32 = -9;
	c13 = +4;  c23 = +7;  c33 = +10;

	for (i = 1; i < NI - 1; ++i) // 0
	{
		for (j = 1; j < NJ - 1; ++j) // 1
		{
			for (k = 1; k < NK -1; ++k) // 2
			{
				//printf("i:%d\nj:%d\nk:%d\n", i, j, k);
				B[i*(NK * NJ) + j*NK + k] = c11 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]  +  c13 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]
					     +   c21 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]  +  c23 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]
					     +   c31 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]  +  c33 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]
					     +   c12 * A[(i + 0)*(NK * NJ) + (j - 1)*NK + (k + 0)]  +  c22 * A[(i + 0)*(NK * NJ) + (j + 0)*NK + (k + 0)]   
					     +   c32 * A[(i + 0)*(NK * NJ) + (j + 1)*NK + (k + 0)]  +  c11 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k + 1)]  
					     +   c13 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k + 1)]  +  c21 * A[(i - 1)*(NK * NJ) + (j + 0)*NK + (k + 1)]  
					     +   c23 * A[(i + 1)*(NK * NJ) + (j + 0)*NK + (k + 1)]  +  c31 * A[(i - 1)*(NK * NJ) + (j + 1)*NK + (k + 1)]  
					     +   c33 * A[(i + 1)*(NK * NJ) + (j + 1)*NK + (k + 1)];
			}
		}
	}
}


int main(int argc, char* argv[]) 
//int main(void) 
{
	double t_start, t_end;

	DATA_TYPE* A;
	DATA_TYPE* B;
	DATA_TYPE* B_outputFromGpu;
        if(argc==2){
          printf("arg 1 = %s\narg 2 = %s\n", argv[0], argv[1]);
          cpu_offset = atoi(argv[1]);
        }


	A = (DATA_TYPE*)malloc(NI*NJ*NK*sizeof(DATA_TYPE));
	B = (DATA_TYPE*)malloc(NI*NJ*NK*sizeof(DATA_TYPE));
	B_outputFromGpu = (DATA_TYPE*)malloc(NI*NJ*NK*sizeof(DATA_TYPE));

	int i;
	init(A);
	read_cl_file();
	cl_initialization_fusion();
	//cl_initialization();
	cl_mem_init(A, B);
	cl_load_prog();

	cl_launch_kernel();

	errcode = clEnqueueReadBuffer(clCommandQue[0], b_mem_obj, CL_TRUE, 0, NI * NJ * NK * sizeof(DATA_TYPE), B_outputFromGpu, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in reading GPU mem\n");

	t_start = rtclock();
	conv3D(A, B);
	t_end = rtclock(); 
	fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);   
	compareResults(B, B_outputFromGpu);
	cl_clean_up();

	free(A);
	free(B);
	free(B_outputFromGpu);

	return 0;
}

