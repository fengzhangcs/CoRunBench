// #ifdef __cplusplus
// extern "C" {
// #endif

//========================================================================================================================================================================================================200
//	DEFINE / INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	COMMON
//======================================================================================================================================================150

#include "../common.h"									// (in path provided here)

//======================================================================================================================================================150
//	KERNEL
//======================================================================================================================================================150

#include "./kernel_fin.c"									// (in path provided here)
#include "../util/opencl/opencl.h"						// (in path provided here)

//======================================================================================================================================================150
//	LIBRARIES
//======================================================================================================================================================150

#include <stdio.h>										// (in path known to compiler)	needed by printf
#include <CL/cl.h>										// (in path provided to compiler)	needed by OpenCL types and functions

//======================================================================================================================================================150
//	END
//======================================================================================================================================================150

//========================================================================================================================================================================================================200
//	MAIN FUNCTION
//========================================================================================================================================================================================================200
extern int cpu_offset;

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

//printf("------>global_work_size=%d\n", global_work_size[0]);
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
  if(global_offset[0]==0)
    cpu_run = 0;
  global_offset_start[0]=global_offset[0];
  global_offset_start[1]=0;
//  printf("gpu: remain_global_work_size=%d\n", remain_global_work_size[0]);
 // printf("cpu: global_offset=%d\n", global_offset[0]);
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


void 
master(	fp timeinst,
		fp *initvalu,
		fp *parameter,
		fp *finavalu,
		fp *com,

		cl_mem d_initvalu,
		cl_mem d_finavalu,
		cl_mem d_params,
		cl_mem d_com,

		cl_command_queue* command_queue,
		//cl_command_queue command_queue,
		cl_kernel kernel,

		long long *timecopyin,
		long long *timekernel,
		long long *timecopyout)
{

	//======================================================================================================================================================150
	//	VARIABLES
	//======================================================================================================================================================150

	//timer
	long long time0;
	long long time1;
	long long time2;
	long long time3;

	// counters
	int i;

	// offset pointers
	int initvalu_offset_ecc;																// 46 points
	int initvalu_offset_Dyad;															// 15 points
	int initvalu_offset_SL;																// 15 points
	int initvalu_offset_Cyt;																// 15 poitns

	// common variables
	cl_int error;

	time0 = get_time();

	//======================================================================================================================================================150
	//	COPY DATA TO GPU MEMORY
	//======================================================================================================================================================150

	//====================================================================================================100
	//	initvalu
	//====================================================================================================100

	int d_initvalu_mem;
	d_initvalu_mem = EQUATIONS * sizeof(fp);
	error = clEnqueueWriteBuffer(	command_queue[0],			// command queue
	//error = clEnqueueWriteBuffer(	command_queue,			// command queue
									d_initvalu,				// destination
									1,						// block the source from access until this copy operation complates (1=yes, 0=no)
									0,						// offset in destination to write to
									d_initvalu_mem,			// size to be copied
									initvalu,				// source
									0,						// # of events in the list of events to wait for
									NULL,					// list of events to wait for
									NULL);					// ID of this operation to be used by waiting operations
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	//	parameter
	//====================================================================================================100

	int d_params_mem;
	d_params_mem = PARAMETERS * sizeof(fp);
	error = clEnqueueWriteBuffer(	command_queue[0],
	//error = clEnqueueWriteBuffer(	command_queue,
									d_params,
									1,
									0,
									d_params_mem,
									parameter,
									0,
									NULL,
									NULL);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	//	END
	//====================================================================================================100

	//time1 = get_time();

	//======================================================================================================================================================150
	//	GPU: KERNEL
	//======================================================================================================================================================150

	//====================================================================================================100
	//	KERNEL EXECUTION PARAMETERS
	//====================================================================================================100

	size_t local_work_size[1];
	local_work_size[0] = NUMBER_THREADS;
	size_t global_work_size[1];
	global_work_size[0] = 2*NUMBER_THREADS;

	// printf("# of blocks = %d, # of threads/block = %d (ensure that device can handle)\n", (int)global_work_size[0]/(int)local_work_size[0], (int)local_work_size[0]);

	//====================================================================================================100
	//	KERNEL ARGUMENTS
	//====================================================================================================100

	clSetKernelArg(	kernel, 
					0, 
					sizeof(int), 
					(void *) &timeinst);
	clSetKernelArg(	kernel, 
					1, 
					sizeof(cl_mem), 
					(void *) &d_initvalu);
	clSetKernelArg(	kernel, 
					2, 
					sizeof(cl_mem), 
					(void *) &d_finavalu);
	clSetKernelArg(	kernel, 
					3, 
					sizeof(cl_mem), 
					(void *) &d_params);
	clSetKernelArg(	kernel, 
					4, 
					sizeof(cl_mem), 
					(void *) &d_com);

	//====================================================================================================100
	//	KERNEL
	//====================================================================================================100

	time1 = get_time();
	error = clEnqueueNDRangeKernel_fusion(	command_queue, 
	//error = clEnqueueNDRangeKernel(	command_queue[0], 
	//error = clEnqueueNDRangeKernel(	command_queue, 
									kernel, 
									1, 
									NULL, 
									global_work_size, 
									local_work_size, 
									0, 
									NULL, 
									NULL);
        /*
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	// Wait for all operations to finish, much like synchronizing threads in CUDA
	error = clFinish(command_queue[0]);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
                */

	time2 = get_time();

	//======================================================================================================================================================150
	//	COPY DATA TO SYSTEM MEMORY
	//======================================================================================================================================================150

	//====================================================================================================100
	//	finavalu
	//====================================================================================================100

	int d_finavalu_mem;
	d_finavalu_mem = EQUATIONS * sizeof(fp);
	error = clEnqueueReadBuffer(command_queue[0],               // The command queue.
	//error = clEnqueueReadBuffer(command_queue,               // The command queue.
								d_finavalu,                  // The image on the device.
								CL_TRUE,                     // Blocking? (ie. Wait at this line until read has finished?)
								0,                           // Offset. None in this case.
								d_finavalu_mem, 			 // Size to copy.
								finavalu,                    // The pointer to the image on the host.
								0,                           // Number of events in wait list. Not used.
								NULL,                        // Event wait list. Not used.
								NULL);                       // Event object for determining status. Not used.
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	//	com
	//====================================================================================================100

	int d_com_mem;
	d_com_mem = 3 * sizeof(fp);
	error = clEnqueueReadBuffer(command_queue[0],
	//error = clEnqueueReadBuffer(command_queue,
								d_com,
								CL_TRUE,
								0,
								d_com_mem,
								com,
								0,
								NULL,
								NULL);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	//	END
	//====================================================================================================100

	time3 = get_time();

	//======================================================================================================================================================150
	//	CPU: FINAL KERNEL
	//======================================================================================================================================================150

	// *copyin_time,
	// *kernel_time,
	// *copyout_time)

	timecopyin[0] = timecopyin[0] + (time1-time0);
	timekernel[0] = timekernel[0] + (time2-time1);
	timecopyout[0] = timecopyout[0] + (time3-time2);

	//======================================================================================================================================================150
	//	CPU: FINAL KERNEL
	//======================================================================================================================================================150

	initvalu_offset_ecc = 0;												// 46 points
	initvalu_offset_Dyad = 46;												// 15 points
	initvalu_offset_SL = 61;												// 15 points
	initvalu_offset_Cyt = 76;												// 15 poitns

	kernel_fin(	initvalu,
				initvalu_offset_ecc,
				initvalu_offset_Dyad,
				initvalu_offset_SL,
				initvalu_offset_Cyt,
				parameter,
				finavalu,
				com[0],
				com[1],
				com[2]);

	//======================================================================================================================================================150
	//	COMPENSATION FOR NANs and INFs
	//======================================================================================================================================================150

	for(i=0; i<EQUATIONS; i++){
		if (isnan(finavalu[i]) == 1){ 
			finavalu[i] = 0.0001;												// for NAN set rate of change to 0.0001
		}
		else if (isinf(finavalu[i]) == 1){ 
			finavalu[i] = 0.0001;												// for INF set rate of change to 0.0001
		}
	}

	//======================================================================================================================================================150
	//	END
	//======================================================================================================================================================150

}

//========================================================================================================================================================================================================200
//	END
//========================================================================================================================================================================================================200

// #ifdef __cplusplus
// }
// #endif