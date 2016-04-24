int cpu_offset;
int tony_device;
#define CEIL(n,d) (n/d+(int)(n%d!=0))
#define Round(n,d) (CEIL(n,d)*d)
void 
fatal_CL(cl_int error, int line_no) {

	printf("Error at line %d: ", line_no);

	switch(error) {

		case CL_SUCCESS: 									printf("CL_SUCCESS\n"); break;
		case CL_DEVICE_NOT_FOUND: 							printf("CL_DEVICE_NOT_FOUND\n"); break;
		case CL_DEVICE_NOT_AVAILABLE: 						printf("CL_DEVICE_NOT_AVAILABLE\n"); break;
		case CL_COMPILER_NOT_AVAILABLE: 					printf("CL_COMPILER_NOT_AVAILABLE\n"); break;
		case CL_MEM_OBJECT_ALLOCATION_FAILURE: 				printf("CL_MEM_OBJECT_ALLOCATION_FAILURE\n"); break;
		case CL_OUT_OF_RESOURCES: 							printf("CL_OUT_OF_RESOURCES\n"); break;
		case CL_OUT_OF_HOST_MEMORY: 						printf("CL_OUT_OF_HOST_MEMORY\n"); break;
		case CL_PROFILING_INFO_NOT_AVAILABLE: 				printf("CL_PROFILING_INFO_NOT_AVAILABLE\n"); break;
		case CL_MEM_COPY_OVERLAP: 							printf("CL_MEM_COPY_OVERLAP\n"); break;
		case CL_IMAGE_FORMAT_MISMATCH: 						printf("CL_IMAGE_FORMAT_MISMATCH\n"); break;
		case CL_IMAGE_FORMAT_NOT_SUPPORTED: 				printf("CL_IMAGE_FORMAT_NOT_SUPPORTED\n"); break;
		case CL_BUILD_PROGRAM_FAILURE: 						printf("CL_BUILD_PROGRAM_FAILURE\n"); break;
		case CL_MAP_FAILURE: 								printf("CL_MAP_FAILURE\n"); break;
		case CL_INVALID_VALUE: 								printf("CL_INVALID_VALUE\n"); break;
		case CL_INVALID_DEVICE_TYPE: 						printf("CL_INVALID_DEVICE_TYPE\n"); break;
		case CL_INVALID_PLATFORM: 							printf("CL_INVALID_PLATFORM\n"); break;
		case CL_INVALID_DEVICE: 							printf("CL_INVALID_DEVICE\n"); break;
		case CL_INVALID_CONTEXT: 							printf("CL_INVALID_CONTEXT\n"); break;
		case CL_INVALID_QUEUE_PROPERTIES: 					printf("CL_INVALID_QUEUE_PROPERTIES\n"); break;
		case CL_INVALID_COMMAND_QUEUE: 						printf("CL_INVALID_COMMAND_QUEUE\n"); break;
		case CL_INVALID_HOST_PTR: 							printf("CL_INVALID_HOST_PTR\n"); break;
		case CL_INVALID_MEM_OBJECT: 						printf("CL_INVALID_MEM_OBJECT\n"); break;
		case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: 			printf("CL_INVALID_IMAGE_FORMAT_DESCRIPTOR\n"); break;
		case CL_INVALID_IMAGE_SIZE: 						printf("CL_INVALID_IMAGE_SIZE\n"); break;
		case CL_INVALID_SAMPLER: 							printf("CL_INVALID_SAMPLER\n"); break;
		case CL_INVALID_BINARY: 							printf("CL_INVALID_BINARY\n"); break;
		case CL_INVALID_BUILD_OPTIONS: 						printf("CL_INVALID_BUILD_OPTIONS\n"); break;
		case CL_INVALID_PROGRAM: 							printf("CL_INVALID_PROGRAM\n"); break;
		case CL_INVALID_PROGRAM_EXECUTABLE: 				printf("CL_INVALID_PROGRAM_EXECUTABLE\n"); break;
		case CL_INVALID_KERNEL_NAME: 						printf("CL_INVALID_KERNEL_NAME\n"); break;
		case CL_INVALID_KERNEL_DEFINITION: 					printf("CL_INVALID_KERNEL_DEFINITION\n"); break;
		case CL_INVALID_KERNEL: 							printf("CL_INVALID_KERNEL\n"); break;
		case CL_INVALID_ARG_INDEX: 							printf("CL_INVALID_ARG_INDEX\n"); break;
		case CL_INVALID_ARG_VALUE: 							printf("CL_INVALID_ARG_VALUE\n"); break;
		case CL_INVALID_ARG_SIZE: 							printf("CL_INVALID_ARG_SIZE\n"); break;
		case CL_INVALID_KERNEL_ARGS: 						printf("CL_INVALID_KERNEL_ARGS\n"); break;
		case CL_INVALID_WORK_DIMENSION: 					printf("CL_INVALID_WORK_DIMENSION\n"); break;
		case CL_INVALID_WORK_GROUP_SIZE: 					printf("CL_INVALID_WORK_GROUP_SIZE\n"); break;
		case CL_INVALID_WORK_ITEM_SIZE: 					printf("CL_INVALID_WORK_ITEM_SIZE\n"); break;
		case CL_INVALID_GLOBAL_OFFSET: 						printf("CL_INVALID_GLOBAL_OFFSET\n"); break;
		case CL_INVALID_EVENT_WAIT_LIST: 					printf("CL_INVALID_EVENT_WAIT_LIST\n"); break;
		case CL_INVALID_EVENT: 								printf("CL_INVALID_EVENT\n"); break;
		case CL_INVALID_OPERATION: 							printf("CL_INVALID_OPERATION\n"); break;
		case CL_INVALID_GL_OBJECT: 							printf("CL_INVALID_GL_OBJECT\n"); break;
		case CL_INVALID_BUFFER_SIZE: 						printf("CL_INVALID_BUFFER_SIZE\n"); break;
		case CL_INVALID_MIP_LEVEL: 							printf("CL_INVALID_MIP_LEVEL\n"); break;
		case CL_INVALID_GLOBAL_WORK_SIZE: 					printf("CL_INVALID_GLOBAL_WORK_SIZE\n"); break;

		#ifdef CL_VERSION_1_1
		case CL_MISALIGNED_SUB_BUFFER_OFFSET: 				printf("CL_MISALIGNED_SUB_BUFFER_OFFSET\n"); break;
		case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:	printf("CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST\n"); break;
		case CL_INVALID_PROPERTY: 							printf("CL_INVALID_PROPERTY\n"); break;
		#endif

		default:											printf("Invalid OpenCL error code\n");

	}

	exit(error);

}
cl_int clEnqueueNDRangeKernel_tony(cl_command_queue * cmdqueue,
                  cl_kernel kernel,cl_uint work_dim,
				  const size_t *global_work_size,const size_t *local_work_size)
  {
	 cl_event eventList[2];
     cl_int error;
     int CPU_RUN=0;
     int GPU_RUN=0;
	if(cpu_offset==0){
     		if(tony_device!=1){
			GPU_RUN=1;
		}	
	}
	else if(cpu_offset==100){
		if(tony_device!=2){
			CPU_RUN=1;
		}
	}
	else{
		if(tony_device==0){
			CPU_RUN=1;	
			GPU_RUN=1;	
		}else if (tony_device==1){
			CPU_RUN=1;
		}else{
			GPU_RUN=1;
		}
	}
     size_t remain_global_work_size[2];
     size_t global_offset[2];
     //NOTES(tony): if dim!=0, offset means offset of first dimensional.
     //we only care for first dimensional.Keep rest remain
         int d;
         for(d=1;d<work_dim;d++){
                 remain_global_work_size[d]=global_work_size[d];
                 global_offset[d]=global_work_size[d];
         }

         global_offset[0]=((double)cpu_offset/100)*(global_work_size[0]);
         global_offset[0]=Round(global_offset[0],(local_work_size[0]));
         remain_global_work_size[0]=global_work_size[0]-global_offset[0];
	 if(remain_global_work_size[0]==0){
	GPU_RUN=0;
	}
         //const size_t *const_remain = remain_global_work_size;
         //printf("global_workSize[0] %d , local_work_size[0] %d, global_offset is %d\n",global_work_size[0],local_work_size    [0],global_offset[0]);

         int numOfArgs;
         clGetKernelInfo(kernel,CL_KERNEL_NUM_ARGS,sizeof(int),&numOfArgs,NULL);
         //printf("Num of args is %d\n", numOfArgs);
         int groupOffset =0;
         if(CPU_RUN){

			//printf("Launch into cpu,globaloffset is %d\n",global_offset[0]);			  
			 clSetKernelArg(kernel, numOfArgs-1, sizeof(int),(void*)&groupOffset);
			 error=clEnqueueNDRangeKernel(cmdqueue[0],kernel,work_dim,NULL,global_offset,local_work_size,0,NULL,&(eventList[0]    ));
			if (error != CL_SUCCESS)
                fatal_CL(error, __LINE__);
         }
    
         if(GPU_RUN){		 
			 groupOffset =global_offset[0] / (local_work_size[0]);
			// printf("Launch into gpu,remain_global_work_sizeis %d\n",remain_global_work_size[0]);	
			 clSetKernelArg(kernel, numOfArgs-1, sizeof(int),(void*)&groupOffset);
			 error =clEnqueueNDRangeKernel(cmdqueue[1],kernel,work_dim,global_offset,remain_global_work_size,local_work_size,0    ,NULL,&(eventList[1]));
			if (error != CL_SUCCESS)
                fatal_CL(error, __LINE__);
         }
   
     if(CPU_RUN)   error|=clFlush(cmdqueue[0]);
     if(GPU_RUN)       error|=clFlush(cmdqueue[1]);
     //  printf("Try to flush\n");
     if(CPU_RUN)       error|=clWaitForEvents(1,&eventList[0]);
     if(GPU_RUN)       error|=clWaitForEvents(1,&eventList[1]);
     //clWaitForEvents(2,eventList);
     //  printf("kernel finished\n");
     return error;
 }
