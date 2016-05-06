/**
 * covariance.c: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
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

/*
#define M 1024
#define N 1024
*/

#define M 2048
#define N 2048

/* Thread block dimensions for kernel 1*/
#define DIM_LOCAL_WORK_GROUP_KERNEL_1_X 64
//#define DIM_LOCAL_WORK_GROUP_KERNEL_1_X 256
#define DIM_LOCAL_WORK_GROUP_KERNEL_1_Y 1

/* Thread block dimensions for kernel 2*/
#define DIM_LOCAL_WORK_GROUP_KERNEL_2_X 8
//#define DIM_LOCAL_WORK_GROUP_KERNEL_2_X 32
#define DIM_LOCAL_WORK_GROUP_KERNEL_2_Y 8

/* Thread block dimensions for kernel 3*/
#define DIM_LOCAL_WORK_GROUP_KERNEL_3_X 64
//#define DIM_LOCAL_WORK_GROUP_KERNEL_3_X 256
#define DIM_LOCAL_WORK_GROUP_KERNEL_3_Y 1

#define sqrt_of_array_cell(x,j) sqrt(x[j])

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

char str_temp[1024];

DATA_TYPE float_n= 3214212.01;
DATA_TYPE eps=  0.005;

cl_platform_id platform_id;
cl_device_id device_id[2];   
//cl_device_id device_id;   
cl_uint num_devices;
cl_uint num_platforms;
cl_int errcode;
cl_context clGPUContext;
cl_kernel clKernel_mean;
cl_kernel clKernel_std;
cl_kernel clKernel_reduce;
cl_kernel clKernel_covar;
cl_command_queue clCommandQue[2];
//cl_command_queue clCommandQue;
cl_program clProgram;
cl_mem data_mem_obj;
cl_mem stddev_mem_obj;
cl_mem mean_mem_obj;
cl_mem symmat_mem_obj;
FILE *fp;
char *source_str;
size_t source_size;

int cpu_offset = 50;


void compareResults(DATA_TYPE* symmat, DATA_TYPE* symmat_outputFromGpu)
{
	int i,j,fail;
	fail = 0;

	for (i=0; i<=M; i++)
	{
		for (j=0; j<=N; j++)
		{
			if (percentDiff(symmat[i*(N+1) + j], symmat_outputFromGpu[i*(N+1) + j]) > PERCENT_DIFF_ERROR_THRESHOLD)
			{
				fail++;
			}			
		}
	}
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);

}


void read_cl_file()
{
	// Load the kernel source code into the array source_str
	fp = fopen("covariance.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );
}


void init_arrays(DATA_TYPE* data)
{
	int i, j;

	for (i = 0; i < M; i++)
	{
		for (j = 0; j < N; j++)
		{
			data[i*(N+1) + j] = ((DATA_TYPE) i*j) / M;
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


void cl_mem_init(DATA_TYPE* data, DATA_TYPE* symmat, DATA_TYPE* mean)
{
	data_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, sizeof(DATA_TYPE) * (M+1) * (N+1), NULL, &errcode);
	symmat_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, sizeof(DATA_TYPE) * (M+1) * (N+1), NULL, &errcode);
	mean_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, sizeof(DATA_TYPE) * (M+1), NULL, &errcode);
		
	if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");

	errcode = clEnqueueWriteBuffer(clCommandQue[0], data_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * (M+1) * (N+1), data, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue[0], symmat_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * (M+1) * (N+1), symmat, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue[0], mean_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * (M+1), mean, 0, NULL, NULL);
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
	clKernel_mean = clCreateKernel(clProgram, "mean_kernel", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel1\n");

	clKernel_reduce = clCreateKernel(clProgram, "reduce_kernel", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel2\n");

	clKernel_covar = clCreateKernel(clProgram, "covar_kernel", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel3\n");
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

	int m = M;
	int n = N;

	size_t localWorkSize_Kernel1[2], globalWorkSize_Kernel1[2];
	size_t localWorkSize_Kernel2[2], globalWorkSize_Kernel2[2];
	size_t localWorkSize_Kernel3[2], globalWorkSize_Kernel3[2];

	localWorkSize_Kernel1[0] = DIM_LOCAL_WORK_GROUP_KERNEL_1_X;
	localWorkSize_Kernel1[1] = DIM_LOCAL_WORK_GROUP_KERNEL_1_Y;
	globalWorkSize_Kernel1[0] = (size_t)ceil(((float)M) / ((float)DIM_LOCAL_WORK_GROUP_KERNEL_1_X)) * DIM_LOCAL_WORK_GROUP_KERNEL_1_X;
	globalWorkSize_Kernel1[1] = 1;

	localWorkSize_Kernel2[0] = DIM_LOCAL_WORK_GROUP_KERNEL_2_X;
	localWorkSize_Kernel2[1] = DIM_LOCAL_WORK_GROUP_KERNEL_2_Y;
	globalWorkSize_Kernel2[0] = (size_t)ceil(((float)M) / ((float)DIM_LOCAL_WORK_GROUP_KERNEL_2_X)) * DIM_LOCAL_WORK_GROUP_KERNEL_2_X;
	globalWorkSize_Kernel2[1] = (size_t)ceil(((float)N) / ((float)DIM_LOCAL_WORK_GROUP_KERNEL_2_Y)) * DIM_LOCAL_WORK_GROUP_KERNEL_2_Y;

	localWorkSize_Kernel3[0] = DIM_LOCAL_WORK_GROUP_KERNEL_3_X;
	localWorkSize_Kernel3[1] = DIM_LOCAL_WORK_GROUP_KERNEL_3_Y;
	globalWorkSize_Kernel3[0] = (size_t)ceil(((float)M) / ((float)DIM_LOCAL_WORK_GROUP_KERNEL_3_X)) * DIM_LOCAL_WORK_GROUP_KERNEL_3_X;
	globalWorkSize_Kernel3[1] = 1;

	t_start = rtclock();
	
	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel_mean, 0, sizeof(cl_mem), (void *)&mean_mem_obj);
	errcode |= clSetKernelArg(clKernel_mean, 1, sizeof(cl_mem), (void *)&data_mem_obj);
	errcode |= clSetKernelArg(clKernel_mean, 2, sizeof(DATA_TYPE), (void *)&float_n);
	errcode |= clSetKernelArg(clKernel_mean, 3, sizeof(int), (void *)&m);
	errcode |= clSetKernelArg(clKernel_mean, 4, sizeof(int), (void *)&n);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments1\n");
	
	// Execute the OpenCL kernel
	//errcode = clEnqueueNDRangeKernel_fusion(clCommandQue, clKernel_mean, 1, NULL, globalWorkSize_Kernel1, localWorkSize_Kernel1, 0, NULL, NULL);
	errcode = clEnqueueNDRangeKernel(clCommandQue[0], clKernel_mean, 1, NULL, globalWorkSize_Kernel1, localWorkSize_Kernel1, 0, NULL, NULL);
        clFinish(clCommandQue[0]);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel1\n");
//	clEnqueueBarrier(clCommandQue[0]);

		
	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel_reduce, 0, sizeof(cl_mem), (void *)&mean_mem_obj);
	errcode |= clSetKernelArg(clKernel_reduce, 1, sizeof(cl_mem), (void *)&data_mem_obj);
	errcode |= clSetKernelArg(clKernel_reduce, 2, sizeof(int), (void *)&m);
	errcode |= clSetKernelArg(clKernel_reduce, 3, sizeof(int), (void *)&n);

	if(errcode != CL_SUCCESS) printf("Error in seting arguments2\n");

	// Execute the OpenCL kernel
	//errcode = clEnqueueNDRangeKernel_fusion(clCommandQue, clKernel_reduce, 2, NULL, globalWorkSize_Kernel2, localWorkSize_Kernel2, 0, NULL, NULL);
	errcode = clEnqueueNDRangeKernel(clCommandQue[0], clKernel_reduce, 2, NULL, globalWorkSize_Kernel2, localWorkSize_Kernel2, 0, NULL, NULL);
        clFinish(clCommandQue[0]);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel1\n");
	if(errcode != CL_SUCCESS) printf("Error in launching kernel2\n");
	//clEnqueueBarrier(clCommandQue[0]);
	
	// Set the arguments of the kernel
	
	errcode =  clSetKernelArg(clKernel_covar, 0, sizeof(cl_mem), (void *)&symmat_mem_obj);
	errcode |= clSetKernelArg(clKernel_covar, 1, sizeof(cl_mem), (void *)&data_mem_obj);
	errcode |= clSetKernelArg(clKernel_covar, 2, sizeof(int), (void *)&m);
	errcode |= clSetKernelArg(clKernel_covar, 3, sizeof(int), (void *)&n);

	if(errcode != CL_SUCCESS) printf("Error in seting arguments4\n");

	// Execute the OpenCL kernel
	errcode = clEnqueueNDRangeKernel_fusion(clCommandQue, clKernel_covar, 1, NULL, globalWorkSize_Kernel3, localWorkSize_Kernel3, 0, NULL, NULL);
	//errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel_covar, 1, NULL, globalWorkSize_Kernel3, localWorkSize_Kernel3, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel4\n");
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
	errcode = clReleaseKernel(clKernel_reduce);
	errcode = clReleaseKernel(clKernel_mean);
	errcode = clReleaseKernel(clKernel_std);
	errcode = clReleaseKernel(clKernel_covar);
	errcode = clReleaseProgram(clProgram);
	errcode = clReleaseMemObject(symmat_mem_obj);
	errcode = clReleaseMemObject(data_mem_obj);
	errcode = clReleaseMemObject(mean_mem_obj);
	errcode = clReleaseMemObject(stddev_mem_obj);
	errcode = clReleaseCommandQueue(clCommandQue[0]);
	errcode = clReleaseCommandQueue(clCommandQue[1]);
	errcode = clReleaseContext(clGPUContext);
	if(errcode != CL_SUCCESS) printf("Error in cleanup\n");
}


void covariance(DATA_TYPE* data, DATA_TYPE* symmat, DATA_TYPE* mean)
{
	int i, j, j1,j2;

  	/* Determine mean of column vectors of input data matrix */
	for (j = 1; j <= M; j++)
	{
		mean[j] = 0.0;
		for (i = 1; i <= N; i++)
		{
        		mean[j] += data[i*(M+1) + j];
		}
		mean[j] /= float_n;
	}

  	/* Center the column vectors. */
	for (i = 1; i <= N; i++)
	{
		for (j = 1; j <= M; j++)
		{
			data[i*(M+1) + j] -= mean[j];
		}
	}

  	/* Calculate the m * m covariance matrix. */
	for (j1 = 1; j1 <= M; j1++)
	{
		for (j2 = j1; j2 <= M; j2++)
   		{
       		symmat[j1*(M+1) + j2] = 0.0;
			for (i = 1; i <= N; i++)
			{
				symmat[j1*(M+1) + j2] += data[i*(M+1) + j1] * data[i*(M+1) + j2];
			}
       		symmat[j2*(M+1) + j1] = symmat[j1*(M+1) + j2];
      	}
	}
}


int main(int argc, char* argv[]) 
//int main(void) 
{
	double t_start, t_end;
	
	DATA_TYPE* data;
	DATA_TYPE* symmat;
	DATA_TYPE* mean;
	DATA_TYPE* symmat_outputFromGpu;	

	DATA_TYPE* stddev_deb;
	DATA_TYPE* mean_deb;
	DATA_TYPE* data_deb;
        if(argc==2){
          printf("arg 1 = %s\narg 2 = %s\n", argv[0], argv[1]);
          cpu_offset = atoi(argv[1]);
        }


	data = (DATA_TYPE*)malloc((M + 1)*(N + 1)*sizeof(DATA_TYPE));
	symmat = (DATA_TYPE*)malloc((M + 1)*(M + 1)*sizeof(DATA_TYPE));
	mean = (DATA_TYPE*)malloc((M + 1)*sizeof(DATA_TYPE));
	symmat_outputFromGpu = (DATA_TYPE*)malloc((M + 1)*(M + 1)*sizeof(DATA_TYPE));	

	init_arrays(data);
	read_cl_file();
	cl_initialization_fusion();
	//cl_initialization();
	cl_mem_init(data, symmat, mean);
	cl_load_prog();

	cl_launch_kernel();

	errcode = clEnqueueReadBuffer(clCommandQue[0], symmat_mem_obj, CL_TRUE, 0, (M+1) * (N+1) * sizeof(DATA_TYPE), symmat_outputFromGpu, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in reading GPU mem\n");   

	t_start = rtclock();
	covariance(data, symmat, mean);
	t_end = rtclock(); 
	fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);   
	compareResults(symmat, symmat_outputFromGpu);
	cl_clean_up();
	
	free(data);
	free(symmat);
	free(mean);
	free(symmat_outputFromGpu);	
	
    return 0;
}

