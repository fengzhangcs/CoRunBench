#ifndef __GAUSSIAN_ELIMINATION__
#define __GAUSSIAN_ELIMINATION__

#include "gaussianElim.h"
#include <math.h>

#ifdef RD_WG_SIZE_0_0
        #define BLOCK_SIZE_0 RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
        #define BLOCK_SIZE_0 RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
        #define BLOCK_SIZE_0 RD_WG_SIZE
#else
        #define BLOCK_SIZE_0 0
#endif

//2D defines. Go from specific to general                                                
#ifdef RD_WG_SIZE_1_0
        #define BLOCK_SIZE_1_X RD_WG_SIZE_1_0
#elif defined(RD_WG_SIZE_1)
        #define BLOCK_SIZE_1_X RD_WG_SIZE_1
#elif defined(RD_WG_SIZE)
        #define BLOCK_SIZE_1_X RD_WG_SIZE
#else
        #define BLOCK_SIZE_1_X 0
#endif

#ifdef RD_WG_SIZE_1_1
        #define BLOCK_SIZE_1_Y RD_WG_SIZE_1_1
#elif defined(RD_WG_SIZE_1)
        #define BLOCK_SIZE_1_Y RD_WG_SIZE_1
#elif defined(RD_WG_SIZE)
        #define BLOCK_SIZE_1_Y RD_WG_SIZE
#else
        #define BLOCK_SIZE_1_Y 0
#endif
int cpu_offset;
extern cl_command_queue commandQueue[2];
cl_context context=NULL;

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
  int errcode ;

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
  int t1=global_offset[0],t2;
  if(local_work_size != NULL){
     t2=local_work_size[0];
     global_offset[0]=(t1/t2+(int)(t1%t2!=0))*t2;
  }
  
//  printf("global_offset[0]=%d\n", (t1/t2+(int)(t1%t2!=0))*t2);
//  printf("t1=%d, t2=%d\n",t1,t2);
  remain_global_work_size[0] = global_work_size[0]-global_offset[0];
  if(remain_global_work_size[0] == 0)
    gpu_run = 0;
  global_offset_start[0]=global_offset[0];
  global_offset_start[1]=0;
//  printf("global_work_size[0]=%d\n", global_work_size[0]);
//exit(-1);
//printf("gpu: dim=%d, remain_global_work_size=%d\n",work_dim, remain_global_work_size[0]);
//printf("gpu: dim=%d, remain_global_work_size=%d, \n",work_dim, remain_global_work_size[0]);
//printf("cpu: dim=%d, global_offset=%d, \n",work_dim, global_offset[0]);
//printf("gpu: dim=%d, remain_global_work_size=%d, local_work_size=%d\n",work_dim, remain_global_work_size[0], local_work_size[0]);
//printf("cpu: dim=%d, global_offset=%d, local_work_size=%d\n",work_dim, global_offset[0], local_work_size[0]);
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


// create both matrix and right hand side, Ke Wang 2013/08/12 11:51:06
void
create_matrix(float *m, int size){
  int i,j;
  float lamda = -0.01;
  float coe[2*size-1];
  float coe_i =0.0;

  for (i=0; i < size; i++)
    {
      coe_i = 10*exp(lamda*i); 
      j=size-1+i;     
      coe[j]=coe_i;
      j=size-1-i;     
      coe[j]=coe_i;
    }


  for (i=0; i < size; i++) {
      for (j=0; j < size; j++) {
	m[i*size+j]=coe[size-1-i+j];
      }
  }


}


int main(int argc, char *argv[]) {

  printf("WG size of kernel 1 = %d, WG size of kernel 2= %d X %d\n", BLOCK_SIZE_0, BLOCK_SIZE_1_X, BLOCK_SIZE_1_Y);
    float *a=NULL, *b=NULL, *finalVec=NULL;
    float *m=NULL;
    int size = -1;
    
    FILE *fp;
    
    // args
    char filename[200];
    int quiet=1,timing=0,platform=-1,device=-1;
    
    // parse command line
    if (parseCommandline(argc, argv, filename,
			 &quiet, &timing, &platform, &device, &size)) {
    printUsage();
    return 0;
    }

    context = cl_init('a');
    //context = cl_init_context(platform,device,quiet);
    
    if(size < 1)
      {
	fp = fopen(filename, "r");
	fscanf(fp, "%d", &size);
    
	a = (float *) malloc(size * size * sizeof(float));
	InitMat(fp,size, a, size, size);

	b = (float *) malloc(size * sizeof(float));
	InitAry(fp, b, size);

	fclose(fp);

      }
    else
      {
	printf("create input internally before create, size = %d \n", size);

	a = (float *) malloc(size * size * sizeof(float));
	create_matrix(a, size);

	b = (float *) malloc(size * sizeof(float));
	for (int i =0; i< size; i++)
	  b[i]=1.0;

      }

    if (!quiet) {    
      printf("The input matrix a is:\n");
      PrintMat(a, size, size, size);

      printf("The input array b is:\n");
      PrintAry(b, size);
    }
 
    // create the solution matrix
    m = (float *) malloc(size * size * sizeof(float));
	 
    // create a new vector to hold the final answer

    finalVec = (float *) malloc(size * sizeof(float));
    
    InitPerRun(size,m);

    //begin timing	
        // printf("The result of array b is before run: \n");
        // PrintAry(b, size);
    
    // run kernels
	ForwardSub(context,a,b,m,size,timing);
        // printf("The result of array b is after run: \n");
        // PrintAry(b, size);
    
    //end timing
    if (!quiet) {
        printf("The result of matrix m is: \n");
        
        PrintMat(m, size, size, size);
        printf("The result of matrix a is: \n");
        PrintMat(a, size, size, size);
        printf("The result of array b is: \n");
        PrintAry(b, size);
        
        BackSub(a,b,finalVec,size);
        printf("The final solution is: \n");
        PrintAry(finalVec,size);
    }
        BackSub(a,b,finalVec,size);
    fp=fopen("result.txt", "w");
    for(int i=0; i<size; i++)
      fprintf(fp,"%f\n", finalVec[i]); 
    fclose(fp);
    
    free(m);
    free(a);
    free(b);
    free(finalVec);
    cl_cleanup();
  //OpenClGaussianElimination(context,timing);

  return 0;
}

double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

/*------------------------------------------------------
 ** ForwardSub() -- Forward substitution of Gaussian
 ** elimination.
 **------------------------------------------------------
 */
void ForwardSub(cl_context context, float *a, float *b, float *m, int size,int timing){    
    // 1. set up kernels
    cl_kernel fan1_kernel,fan2_kernel;
    cl_int status=0;
    cl_program gaussianElim_program;
    cl_event writeEvent,kernelEvent,readEvent;
    float writeTime=0,readTime=0,kernelTime=0;
    float writeMB=0,readMB=0;
    
    gaussianElim_program = cl_compileProgram(
        (char *)"gaussianElim_kernels.cl",NULL);
   
    fan1_kernel = clCreateKernel(
        gaussianElim_program, "Fan1", &status);
    status = cl_errChk(status, (char *)"Error Creating Fan1 kernel",true);
    if(status)exit(1);
   
    fan2_kernel = clCreateKernel(
        gaussianElim_program, "Fan2", &status);
    status = cl_errChk(status, (char *)"Error Creating Fan2 kernel",true);
    if(status)exit(1);
    
    // 2. set up memory on device and send ipts data to device

    cl_mem a_dev, b_dev, m_dev;

    cl_int error=0;

    a_dev = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR,
        sizeof(float)*size*size, NULL, &error);
        
    b_dev = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR,
        sizeof(float)*size, NULL, &error);

    m_dev = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR,
        sizeof(float) * size * size, NULL, &error);

    cl_command_queue command_queue[2];
    command_queue[0]=commandQueue[0];
    command_queue[1]=commandQueue[1];
    //cl_command_queue command_queue = cl_getCommandQueue();
    
    error = clEnqueueWriteBuffer(command_queue[0],
               a_dev,
               1, // change to 0 for nonblocking write
               0, // offset
               sizeof(float)*size*size,
               a,
               0,
               NULL,
               &writeEvent);
    
    if (timing) writeTime+=eventTime(writeEvent,command_queue[0]);
    clReleaseEvent(writeEvent);
    
    error = clEnqueueWriteBuffer(command_queue[0],
               b_dev,
               1, // change to 0 for nonblocking write
               0, // offset
               sizeof(float)*size,
               b,
               0,
               NULL,
               &writeEvent);
    if (timing) writeTime+=eventTime(writeEvent,command_queue[0]);
    clReleaseEvent(writeEvent);
               
    error = clEnqueueWriteBuffer(command_queue[0],
               m_dev,
               1, // change to 0 for nonblocking write
               0, // offset
               sizeof(float)*size*size,
               m,
               0,
               NULL,
               &writeEvent);
    if (timing) writeTime+=eventTime(writeEvent,command_queue[0]);
    clReleaseEvent(writeEvent);
    writeMB = (float)(sizeof(float) * size * (size + size + 1) / 1e6);

    // 3. Determine block sizes
    size_t globalWorksizeFan1[1];
    size_t globalWorksizeFan2[2];
    size_t localWorksizeFan1Buf[1]={BLOCK_SIZE_0};
    size_t localWorksizeFan2Buf[2]={BLOCK_SIZE_1_X, BLOCK_SIZE_1_Y};
    size_t *localWorksizeFan1=NULL;
    size_t *localWorksizeFan2=NULL;

        globalWorksizeFan1[0] = size;
        globalWorksizeFan2[0] = size;
        globalWorksizeFan2[1] = size;

        if(localWorksizeFan1Buf[0]){
                localWorksizeFan1=localWorksizeFan1Buf;
                globalWorksizeFan1[0]=(int)ceil(globalWorksizeFan1[0]/(double)localWorks\
izeFan1Buf[0])*localWorksizeFan1Buf[0];
        }
        if(localWorksizeFan2Buf[0]){
                localWorksizeFan2=localWorksizeFan2Buf;
                globalWorksizeFan2[0]=(int)ceil(globalWorksizeFan2[0]/(double)localWorks\
izeFan2Buf[0])*localWorksizeFan2Buf[0];
                globalWorksizeFan2[1]=(int)ceil(globalWorksizeFan2[1]/(double)localWorks\
izeFan2Buf[1])*localWorksizeFan2Buf[1];
        }

	int t;
	// 4. Setup and Run kernels
      double start = gettime();
      for (t=0; t<(size-1); t++) {
        // kernel args
        cl_int argchk;
        argchk  = clSetKernelArg(fan1_kernel, 0, sizeof(cl_mem), (void *)&m_dev);
        argchk |= clSetKernelArg(fan1_kernel, 1, sizeof(cl_mem), (void *)&a_dev);
        argchk |= clSetKernelArg(fan1_kernel, 2, sizeof(cl_mem), (void *)&b_dev);
        argchk |= clSetKernelArg(fan1_kernel, 3, sizeof(int), (void *)&size);
        argchk |= clSetKernelArg(fan1_kernel, 4, sizeof(int), (void *)&t);
    
        cl_errChk(argchk,"ERROR in Setting Fan1 kernel args",true);
        
        // launch kernel
//printf("hahah globalWorksizeFan1=%d, localWorksizeFan1=%d\n", globalWorksizeFan1[0], localWorksizeFan1[0]);//exit(-1);
        error = clEnqueueNDRangeKernel(
                  command_queue[0],  fan1_kernel, 1, 0,
        //error = clEnqueueNDRangeKernel_fusion(
          //        command_queue,  fan1_kernel, 1, 0,
                  globalWorksizeFan1,localWorksizeFan1,
                  0, NULL, &kernelEvent);

        cl_errChk(error,"ERROR in Executing Fan1 Kernel",true);
        if (timing) {
//             printf("here1a\n");
             kernelTime+=eventTime(kernelEvent,command_queue[0]);
//             printf("here1b\n");
        }
  //zf      clReleaseEvent(kernelEvent);
//printf("hahah globalWorksizeFan1=%d\n", globalWorksizeFan1[0]);exit(-1);

		//Fan1<<<dimGrid,dimBlock>>>(m_cuda,a_cuda,Size,t);
		//cudaThreadSynchronize();
		
		// kernel args
		argchk  = clSetKernelArg(fan2_kernel, 0, sizeof(cl_mem), (void *)&m_dev);
        argchk |= clSetKernelArg(fan2_kernel, 1, sizeof(cl_mem), (void *)&a_dev);
        argchk |= clSetKernelArg(fan2_kernel, 2, sizeof(cl_mem), (void *)&b_dev);
        argchk |= clSetKernelArg(fan2_kernel, 3, sizeof(int), (void *)&size);
        argchk |= clSetKernelArg(fan2_kernel, 4, sizeof(int), (void *)&t);
    
        cl_errChk(argchk,"ERROR in Setting Fan2 kernel args",true);
        
        // launch kernel
        error = clEnqueueNDRangeKernel_fusion(
                  command_queue,  fan2_kernel, 2, 0,
        //error = clEnqueueNDRangeKernel(
          //        command_queue[0],  fan2_kernel, 2, 0,
                  globalWorksizeFan2,localWorksizeFan2,
                  //globalWorksizeFan2,NULL,
                  0, NULL, &kernelEvent);

        cl_errChk(error,"ERROR in Executing Fan1 Kernel",true);
        if (timing) {
//             printf("here2a\n");
             kernelTime+=eventTime(kernelEvent,command_queue[0]);
//             printf("here2b\n");
        }
//        clReleaseEvent(kernelEvent);
		//Fan2<<<dimGridXY,dimBlockXY>>>(m_cuda,a_cuda,b_cuda,Size,Size-t,t);
		//cudaThreadSynchronize();
      }
      double end = gettime();
        printf("CAUTION: cpu_offset: %d time: %lf mseconds\n", cpu_offset, 1000*(end-start));
    // 5. transfer data off of device
    error = clEnqueueReadBuffer(command_queue[0],
        a_dev,
        1, // change to 0 for nonblocking write
        0, // offset
        sizeof(float) * size * size,
        a,
        0,
        NULL,
        &readEvent);

    cl_errChk(error,"ERROR with clEnqueueReadBuffer",true);
    if (timing) readTime+=eventTime(readEvent,command_queue[0]);
    clReleaseEvent(readEvent);
    
    error = clEnqueueReadBuffer(command_queue[0],
        b_dev,
        1, // change to 0 for nonblocking write
        0, // offset
        sizeof(float) * size,
        b,
        0,
        NULL,
        &readEvent);
    cl_errChk(error,"ERROR with clEnqueueReadBuffer",true);
    if (timing) readTime+=eventTime(readEvent,command_queue[0]);
    clReleaseEvent(readEvent);
    
    error = clEnqueueReadBuffer(command_queue[0],
        m_dev,
        1, // change to 0 for nonblocking write
        0, // offset
        sizeof(float) * size * size,
        m,
        0,
        NULL,
        &readEvent);

    cl_errChk(error,"ERROR with clEnqueueReadBuffer",true);
    if (timing) readTime+=eventTime(readEvent,command_queue[0]);
    clReleaseEvent(readEvent);
    readMB = (float)(sizeof(float) * size * (size + size + 1) / 1e6);
    
    if (timing) {
        printf("Matrix Size\tWrite(s) [size]\t\tKernel(s)\tRead(s)  [size]\t\tTotal(s)\n");
        printf("%dx%d      \t",size,size);
        
        printf("%f [%.2fMB]\t",writeTime,writeMB);
        

        printf("%f\t",kernelTime);
       

        printf("%f [%.2fMB]\t",readTime,readMB);
       
        printf("%f\n\n",writeTime+kernelTime+readTime);
   }
    
}

float eventTime(cl_event event,cl_command_queue command_queue){
    cl_int error=0;
    cl_ulong eventStart,eventEnd;
    clFinish(command_queue);
    error = clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,
                                    sizeof(cl_ulong),&eventStart,NULL);
    cl_errChk(error,"ERROR in Event Profiling.",true); 
    error = clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,
                                    sizeof(cl_ulong),&eventEnd,NULL);
    cl_errChk(error,"ERROR in Event Profiling.",true);

    return (float)((eventEnd-eventStart)/1e9);
}

 // Ke Wang add a function to generate input internally
int parseCommandline(int argc, char *argv[], char* filename,
                     int *q, int *t, int *p, int *d, int *size){
    int i;
    if (argc < 2) return 1; // error
    // strncpy(filename,argv[1],100);
    char flag;
    
    for(i=1;i<argc;i++) {
      if (argv[i][0]=='-') {// flag
        flag = argv[i][1];
          switch (flag) {
            case 's': // platform
              i++;
              *size = atoi(argv[i]);
	      printf("Create matrix internally in parse, size = %d \n", *size);
              break;
            case 'f': // platform
              i++;
	      strncpy(filename,argv[i],100);
	      printf("Read file from %s \n", filename);
              break;
            case 'h': // help
              return 1;
              break;
            case 'q': // quiet
              *q = 1;
              break;
            case 't': // timing
              *t = 1;
              break;
            case 'p': // platform
              i++;
              *p = atoi(argv[i]);
              break;
            case 'd': // device
              i++;
              *d = atoi(argv[i]);
              break;
            case 'a':
              i++;
              cpu_offset=atoi(argv[i]);
              printf("cpuoffset = %d\n", cpu_offset);
              break;
        }
      }
    }
    if ((*d >= 0 && *p<0) || (*p>=0 && *d<0)) // both p and d must be specified if either are specified
      return 1;
    return 0;
}

void printUsage(){
  printf("Gaussian Elimination Usage\n");
  printf("\n");
  printf("gaussianElimination [filename] [-hqt] [-p [int] -d [int]]\n");
  printf("\n");
  printf("example:\n");
  printf("$ ./gaussianElimination matrix4.txt\n");
  printf("\n");
  printf("filename     the filename that holds the matrix data\n");
  printf("\n");
  printf("-h           Display the help file\n");
  printf("-q           Quiet mode. Suppress all text output.\n");
  printf("-t           Print timing information.\n");
  printf("\n");
  printf("-p [int]     Choose the platform (must choose both platform and device)\n");
  printf("-d [int]     Choose the device (must choose both platform and device)\n");
  printf("\n");
  printf("\n");
  printf("Notes: 1. The filename is required as the first parameter.\n");
  printf("       2. If you declare either the device or the platform,\n");
  printf("          you must declare both.\n\n");
}


/*------------------------------------------------------
 ** InitPerRun() -- Initialize the contents of the
 ** multipier matrix **m
 **------------------------------------------------------
 */
void InitPerRun(int size,float *m) 
{
	int i;
	for (i=0; i<size*size; i++)
			*(m+i) = 0.0;
}
void BackSub(float *a, float *b, float *finalVec, int size)
{
	// solve "bottom up"
	int i,j;
	for(i=0;i<size;i++){
		finalVec[size-i-1]=b[size-i-1];
		for(j=0;j<i;j++)
		{
			finalVec[size-i-1]-=*(a+size*(size-i-1)+(size-j-1)) * finalVec[size-j-1];
		}
		finalVec[size-i-1]=finalVec[size-i-1]/ *(a+size*(size-i-1)+(size-i-1));
	}
}
void InitMat(FILE *fp, int size, float *ary, int nrow, int ncol)
{
	int i, j;
	
	for (i=0; i<nrow; i++) {
		for (j=0; j<ncol; j++) {
			fscanf(fp, "%f",  ary+size*i+j);
		}
	}  
}
/*------------------------------------------------------
 ** InitAry() -- Initialize the array (vector) by reading
 ** data from the data file
 **------------------------------------------------------
 */
void InitAry(FILE *fp, float *ary, int ary_size)
{
	int i;
	
	for (i=0; i<ary_size; i++) {
		fscanf(fp, "%f",  &ary[i]);
	}
}  
/*------------------------------------------------------
 ** PrintMat() -- Print the contents of the matrix
 **------------------------------------------------------
 */
void PrintMat(float *ary, int size, int nrow, int ncol)
{
	int i, j;
	
	for (i=0; i<nrow; i++) {
		for (j=0; j<ncol; j++) {
			printf("%8.2e ", *(ary+size*i+j));
		}
		printf("\n");
	}
	printf("\n");
}

/*------------------------------------------------------
 ** PrintAry() -- Print the contents of the array (vector)
 **------------------------------------------------------
 */
void PrintAry(float *ary, int ary_size)
{
	int i;
	for (i=0; i<ary_size; i++) {
		printf("%.2e ", ary[i]);
	}
	printf("\n\n");
}
#endif

