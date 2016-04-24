#include"fusion.h"
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

  clFinish(command_queue[0]);
  clFinish(command_queue[1]);

  cl_event eventList[2];
  int cpu_run=0, gpu_run=0;
 size_t global_offset[3];
 size_t global_offset_start[3];
 size_t remain_global_work_size[3];
 int i;
  int errcode ;
//  printf("work dim = %d\n", work_dim);

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
  if(global_offset[0] == 0)
    cpu_run = 0;
  global_offset_start[0]=global_offset[0];
  global_offset_start[1]=0;
  global_offset_start[2]=0;

 /* 
  printf("\nlocal_work_size[0]=%d, local_work_size[1]=%d\n", local_work_size[0], local_work_size[1]);
  printf("total: global_work_size[0]=%d cpu:%d%\n", global_work_size[0], cpu_offset);
  printf("cpu: global_offset[0]=%d, global_offset[1]=%d\n", global_offset[0],global_offset[1]);
  printf("gpu: remain_global_work_size[0]=%d, remain_global_work_size[1]=%d\n", remain_global_work_size[0],remain_global_work_size[1]);
  printf("gpu: global_offset_start[0]=%d\n", global_offset_start[0]);
  */
  

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



