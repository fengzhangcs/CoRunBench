/*
 * =====================================================================================
 *
 *       Filename:  lud.cu
 *
 *    Description:  The main wrapper for the suite
 *
 *        Version:  1.0
 *        Created:  10/22/2009 08:40:34 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Liang Wang (lw2aw), lw2aw@virginia.edu
 *        Company:  CS@UVa
 *
 * =====================================================================================
 */

#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>
#include <assert.h>

#include "common.h"
#include <sys/time.h>
#include <CL/cl.h>

#include <string.h>
#include <string>
#ifdef RD_WG_SIZE_0_0
        #define BLOCK_SIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
        #define BLOCK_SIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
        #define BLOCK_SIZE RD_WG_SIZE
#else
        #define BLOCK_SIZE 16
#endif

 double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

static cl_context	    context;
static cl_command_queue cmd_queue[2];//zf
//static cl_command_queue cmd_queue;
static cl_device_type   device_type;
static cl_device_id    device_list[2];//zf
//static cl_device_id   * device_list;
static cl_uint           num_devices;

int cpu_offset;
cl_int errcode;

cl_int clEnqueueNDRangeKernel_fusion ( cl_command_queue* command_queue,
    cl_kernel kernel,
    cl_uint work_dim,
    const size_t *global_work_offset,
    const size_t *global_work_size,
    const size_t *local_work_size,
    cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list,
    cl_event *event){

  clFinish(command_queue[0]);
  clFinish(command_queue[1]);

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



static int initialize_fusion()
{
	cl_int result;
	size_t size;

	// create OpenCL context
	cl_platform_id platform_id;
	if (clGetPlatformIDs(1, &platform_id, NULL) != CL_SUCCESS) { 
          printf("ERROR: clGetPlatformIDs(1,*,0) failed\n"); return -1; 
        }


        errcode = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1, &device_list[0], &num_devices);
        if(errcode == CL_SUCCESS) printf("number of GPU is %d\n", num_devices);
        errcode |= clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_CPU, 1, &device_list[1], &num_devices);
        if(errcode == CL_SUCCESS) printf("number of CPU is %d\n", num_devices);
        else printf("Error getting device IDs\n");

        char str_temp[1024];
        errcode = clGetDeviceInfo(device_list[0],CL_DEVICE_NAME, sizeof(str_temp), str_temp,NULL);
        if(errcode == CL_SUCCESS) printf("GPU device name is %s\n",str_temp);
        else printf("GPU Error getting device name\n");
        errcode = clGetDeviceInfo(device_list[1],CL_DEVICE_NAME, sizeof(str_temp), str_temp,NULL);
        if(errcode == CL_SUCCESS) printf("CPU device name is %s\n",str_temp);
        else printf("CPU Error getting device name\n");



	cl_context_properties ctxprop[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id, 0};
//	device_type = use_gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU;
        context = clCreateContext( NULL, 2, device_list, NULL, NULL, NULL);
	//context = clCreateContextFromType( ctxprop, device_type, NULL, NULL, NULL );
	if( !context ) { printf("ERROR: clCreateContextFromType() failed\n" ); return -1; }

        /*
	// get the list of GPUs
	result = clGetContextInfo( context, CL_CONTEXT_DEVICES, 0, NULL, &size );
	num_devices = (int) (size / sizeof(cl_device_id));
	printf("num_devices = %d\n", num_devices);
	
	if( result != CL_SUCCESS || num_devices < 1 ) { printf("ERROR: clGetContextInfo() failed\n"); return -1; }
	device_list = new cl_device_id[num_devices];
	if( !device_list ) { printf("ERROR: new cl_device_id[] failed\n"); return -1; }
	result = clGetContextInfo( context, CL_CONTEXT_DEVICES, size, device_list, NULL );
	if( result != CL_SUCCESS ) { printf("ERROR: clGetContextInfo() failed\n"); return -1; }
        */

	// create command queue for the first device
	cmd_queue[0] = clCreateCommandQueue( context, device_list[0], 0, NULL );
	if( !cmd_queue[0] ) { printf("ERROR: gpu clCreateCommandQueue() failed\n"); return -1; }
	cmd_queue[1] = clCreateCommandQueue( context, device_list[1], 0, NULL );
	if( !cmd_queue[1] ) { printf("ERROR: cpu clCreateCommandQueue() failed\n"); return -1; }
	return 0;
}

/*
static int initialize(int use_gpu)
{
	cl_int result;
	size_t size;

	// create OpenCL context
	cl_platform_id platform_id;
	if (clGetPlatformIDs(1, &platform_id, NULL) != CL_SUCCESS) { printf("ERROR: clGetPlatformIDs(1,*,0) failed\n"); return -1; }
	cl_context_properties ctxprop[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id, 0};
	device_type = use_gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU;
	context = clCreateContextFromType( ctxprop, device_type, NULL, NULL, NULL );
	if( !context ) { printf("ERROR: clCreateContextFromType(%s) failed\n", use_gpu ? "GPU" : "CPU"); return -1; }

	// get the list of GPUs
	result = clGetContextInfo( context, CL_CONTEXT_DEVICES, 0, NULL, &size );
	num_devices = (int) (size / sizeof(cl_device_id));
	printf("num_devices = %d\n", num_devices);
	
	if( result != CL_SUCCESS || num_devices < 1 ) { printf("ERROR: clGetContextInfo() failed\n"); return -1; }
	device_list = new cl_device_id[num_devices];
	if( !device_list ) { printf("ERROR: new cl_device_id[] failed\n"); return -1; }
	result = clGetContextInfo( context, CL_CONTEXT_DEVICES, size, device_list, NULL );
	if( result != CL_SUCCESS ) { printf("ERROR: clGetContextInfo() failed\n"); return -1; }

	// create command queue for the first device
	cmd_queue = clCreateCommandQueue( context, device_list[0], 0, NULL );
	if( !cmd_queue ) { printf("ERROR: clCreateCommandQueue() failed\n"); return -1; }
	return 0;
}
*/

static int shutdown()
{
	// release resources
	if( cmd_queue[1] ) clReleaseCommandQueue( cmd_queue[1] );
	if( cmd_queue[0] ) clReleaseCommandQueue( cmd_queue[0] );
	if( context ) clReleaseContext( context );
//	if( device_list ) delete device_list;

	// reset all variables
	cmd_queue[1] = 0;
	cmd_queue[0] = 0;
	context = 0;
	device_list[0] = 0;
	device_list[1] = 0;
	num_devices = 0;
	device_type = 0;

	return 0;
}

static int do_verify = 0;
void lud_cuda(float *d_m, int matrix_dim);

static struct option long_options[] = {
      /* name, has_arg, flag, val */
      {"input", 1, NULL, 'i'},
      {"size", 1, NULL, 's'},
      {"verify", 0, NULL, 'v'},
      {0,0,0,0}
};

int
main ( int argc, char *argv[] )
{
  printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);
	int matrix_dim = 32; /* default matrix_dim */
	int opt, option_index=0;
	func_ret_t ret;
	const char *input_file = NULL;
	float *m, *mm;
	stopwatch sw;
        cpu_offset=100;
	
	while ((opt = getopt_long(argc, argv, "::vs:i:f:", 
	//while ((opt = getopt_long(argc, argv, "::vs:i:", 
                            long_options, &option_index)) != -1 ) {
		switch(opt){
			case 'i':
			input_file = optarg;
			break;
			case 'v':
			do_verify = 1;
			break;
        case 's':
			matrix_dim = atoi(optarg);
			printf("Generate input matrix internally, size =%d\n", matrix_dim);
			// fprintf(stderr, "Currently not supported, use -i instead\n");
			// fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n", argv[0]);
			// exit(EXIT_FAILURE);
			break;
        case 'f':
                        cpu_offset = atoi(optarg);
                        printf("cpu offset = %d\n", cpu_offset);
			break;

        case '?':
			fprintf(stderr, "invalid option\n");
			break;
        case ':':
			fprintf(stderr, "missing argument\n");
			break;
        default:
			fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n",
                  argv[0]);
			exit(EXIT_FAILURE);
		}
	}
  
	if ( (optind < argc) || (optind == 1)) {
		fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n", argv[0]);
		exit(EXIT_FAILURE);
	}	

	if (input_file) {
		printf("Reading matrix from file %s\n", input_file);
		ret = create_matrix_from_file(&m, input_file, &matrix_dim);
		if (ret != RET_SUCCESS) {
			m = NULL;
			fprintf(stderr, "error create matrix from file %s\n", input_file);
			exit(EXIT_FAILURE);
		}
	} 
	
	else if (matrix_dim) {
	  printf("Creating matrix internally size=%d\n", matrix_dim);
	  ret = create_matrix(&m, matrix_dim);
	  if (ret != RET_SUCCESS) {
	    m = NULL;
	    fprintf(stderr, "error create matrix internally size=%d\n", matrix_dim);
	    exit(EXIT_FAILURE);
	  }
	}

	else {
	  printf("No input file specified!\n");
	  exit(EXIT_FAILURE);
	}

	if (do_verify){
		printf("Before LUD\n");
		// print_matrix(m, matrix_dim);
		matrix_duplicate(m, &mm, matrix_dim);
	}
	
	int sourcesize = 1024*1024;
	char * source = (char *)calloc(sourcesize, sizeof(char)); 
	if(!source) { printf("ERROR: calloc(%d) failed\n", sourcesize); return -1; }

	char * kernel_lud_diag   = "lud_diagonal";
	char * kernel_lud_peri   = "lud_perimeter";
	char * kernel_lud_inter  = "lud_internal";
	FILE * fp = fopen("./lud_kernel.cl", "rb"); 
	if(!fp) { printf("ERROR: unable to open '%s'\n"); return -1; }
	fread(source + strlen(source), sourcesize, 1, fp);
	fclose(fp);

	// Use 1: GPU  0: CPU
	int use_gpu = 1;
	// OpenCL initialization
	if(initialize_fusion()) 
          return -1;
	// compile kernel
	cl_int err = 0;
	const char * slist[2] = { source, 0 };
	cl_program prog = clCreateProgramWithSource(context, 1, slist, NULL, &err);
	if(err != CL_SUCCESS) { printf("ERROR: clCreateProgramWithSource() => %d\n", err); return -1; }
	char clOptions[110];
	//  sprintf(clOptions,"-I../../src"); 
	sprintf(clOptions," ");
#ifdef BLOCK_SIZE
	sprintf(clOptions + strlen(clOptions), " -DBLOCK_SIZE=%d", BLOCK_SIZE);
#endif

	err = clBuildProgram(prog, 0, NULL, clOptions, NULL, NULL);
	{ // show warnings/errors
		//static char log[65536]; memset(log, 0, sizeof(log));
		//cl_device_id device_id = 0;
		//err = clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(device_id), &device_id, NULL);
		//clGetProgramBuildInfo(prog, device_id, CL_PROGRAM_BUILD_LOG, sizeof(log)-1, log, NULL);
		//if(err || strstr(log,"warning:") || strstr(log, "error:")) printf("<<<<\n%s\n>>>>\n", log);
	}
	if(err != CL_SUCCESS) { printf("ERROR: clBuildProgram() => %d\n", err); return -1; }
    
	cl_kernel diagnal;
	cl_kernel perimeter;
	cl_kernel internal;
	diagnal   = clCreateKernel(prog, kernel_lud_diag, &err);  
	perimeter = clCreateKernel(prog, kernel_lud_peri, &err);  
	internal  = clCreateKernel(prog, kernel_lud_inter, &err);  
	if(err != CL_SUCCESS) { printf("ERROR: clCreateKernel() 0 => %d\n", err); return -1; }
	clReleaseProgram(prog);
  
	//size_t local_work[3] = { 1, 1, 1 };
	//size_t global_work[3] = {1, 1, 1 }; 
  
	cl_mem d_m;
	d_m = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, matrix_dim*matrix_dim * sizeof(float), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer d_m (size:%d) => %d\n", matrix_dim*matrix_dim, err); return -1;} 

	/* beginning of timing point */
	//stopwatch_start(&sw);
	err = clEnqueueWriteBuffer(cmd_queue[0], d_m, 1, 0, matrix_dim*matrix_dim*sizeof(float), m, 0, 0, 0);
	if(err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer d_m (size:%d) => %d\n", matrix_dim*matrix_dim, err); return -1; }
	clFinish(cmd_queue[0]);
	stopwatch_start(&sw);
	
	int i=0;
	for (i=0; i < matrix_dim-BLOCK_SIZE; i += BLOCK_SIZE) {
	 
	  clSetKernelArg(diagnal, 0, sizeof(void *), (void*) &d_m);
	  clSetKernelArg(diagnal, 1, sizeof(float) * BLOCK_SIZE * BLOCK_SIZE, (void*)NULL );
	  clSetKernelArg(diagnal, 2, sizeof(cl_int), (void*) &matrix_dim);
	  clSetKernelArg(diagnal, 3, sizeof(cl_int), (void*) &i);
      
	  size_t global_work1[3]  = {BLOCK_SIZE, 1, 1};
	  size_t local_work1[3]  = {BLOCK_SIZE, 1, 1};
	   
	  //err = clEnqueueNDRangeKernel_fusion(cmd_queue, diagnal, 2, NULL, global_work1, local_work1, 0, 0, 0);
	  err = clEnqueueNDRangeKernel(cmd_queue[0], diagnal, 2, NULL, global_work1, local_work1, 0, 0, 0);
	  if(err != CL_SUCCESS) { printf("ERROR:  diagnal clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }	
	  
	  clSetKernelArg(perimeter, 0, sizeof(void *), (void*) &d_m);
	  clSetKernelArg(perimeter, 1, sizeof(float) * BLOCK_SIZE * BLOCK_SIZE, (void*)NULL );
	  clSetKernelArg(perimeter, 2, sizeof(float) * BLOCK_SIZE * BLOCK_SIZE, (void*)NULL );
	  clSetKernelArg(perimeter, 3, sizeof(float) * BLOCK_SIZE * BLOCK_SIZE, (void*)NULL );
	  clSetKernelArg(perimeter, 4, sizeof(cl_int), (void*) &matrix_dim);
	  clSetKernelArg(perimeter, 5, sizeof(cl_int), (void*) &i);
	  
	  size_t global_work2[3] = {BLOCK_SIZE * 2 * ((matrix_dim-i)/BLOCK_SIZE-1), 1, 1};
	  size_t local_work2[3]  = {BLOCK_SIZE * 2, 1, 1};
	  
	  //err = clEnqueueNDRangeKernel_fusion(cmd_queue, perimeter, 2, NULL, global_work2, local_work2, 0, 0, 0);
	  err = clEnqueueNDRangeKernel(cmd_queue[0], perimeter, 2, NULL, global_work2, local_work2, 0, 0, 0);
	  if(err != CL_SUCCESS) { printf("ERROR:  perimeter clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }	
	  
	  clSetKernelArg(internal, 0, sizeof(void *), (void*) &d_m);
	  clSetKernelArg(internal, 1, sizeof(float) * BLOCK_SIZE * BLOCK_SIZE, (void*)NULL );
	  clSetKernelArg(internal, 2, sizeof(float) * BLOCK_SIZE * BLOCK_SIZE, (void*)NULL );
	  clSetKernelArg(internal, 3, sizeof(cl_int), (void*) &matrix_dim);
	  clSetKernelArg(internal, 4, sizeof(cl_int), (void*) &i);
      
	  size_t global_work3[3] = {BLOCK_SIZE * ((matrix_dim-i)/BLOCK_SIZE-1), BLOCK_SIZE * ((matrix_dim-i)/BLOCK_SIZE-1), 1};
	  size_t local_work3[3] = {BLOCK_SIZE, BLOCK_SIZE, 1};

	  err = clEnqueueNDRangeKernel_fusion(cmd_queue, internal, 2, NULL, global_work3, local_work3, 0, 0, 0);
	  //err = clEnqueueNDRangeKernel(cmd_queue[0], internal, 2, NULL, global_work3, local_work3, 0, 0, 0);
	  if(err != CL_SUCCESS) { printf("ERROR:  internal clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }	
	}
	clSetKernelArg(diagnal, 0, sizeof(void *), (void*) &d_m);
	clSetKernelArg(diagnal, 1, sizeof(float) * BLOCK_SIZE * BLOCK_SIZE, (void*)NULL );
	clSetKernelArg(diagnal, 2, sizeof(cl_int), (void*) &matrix_dim);
	clSetKernelArg(diagnal, 3, sizeof(cl_int), (void*) &i);
      
	size_t global_work1[3]  = {BLOCK_SIZE, 1, 1};
	size_t local_work1[3]  = {BLOCK_SIZE, 1, 1};
	//err = clEnqueueNDRangeKernel_fusion(cmd_queue, diagnal, 2, NULL, global_work1, local_work1, 0, 0, 0);
	err = clEnqueueNDRangeKernel(cmd_queue[0], diagnal, 2, NULL, global_work1, local_work1, 0, 0, 0);
	if(err != CL_SUCCESS) { printf("ERROR:  diagnal clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }	
	stopwatch_stop(&sw);
	
	err = clEnqueueReadBuffer(cmd_queue[0], d_m, 1, 0, matrix_dim*matrix_dim*sizeof(float), m, 0, 0, 0);
	if(err != CL_SUCCESS) { printf("ERROR: clEnqueueReadBuffer  d_m (size:%d) => %d\n", matrix_dim*matrix_dim, err); return -1; }
	clFinish(cmd_queue[0]);
	/* end of timing point */
	//stopwatch_stop(&sw);
	printf("Time consumed(ms): %lf\n", 1000*get_interval_by_sec(&sw));

         printf("CAUTION: cpu_offset: %d time: %lf mseconds\n", cpu_offset, 1000*get_interval_by_sec(&sw));

	clReleaseMemObject(d_m);

	if (do_verify){
		printf("After LUD\n");
		// print_matrix(m, matrix_dim);
		printf(">>>Verify<<<<\n");
		lud_verify(mm, m, matrix_dim); 
		free(mm);
	}

	free(m);
	
	if(shutdown()) return -1;
	
}				

/* ----------  end of function main  ---------- */


