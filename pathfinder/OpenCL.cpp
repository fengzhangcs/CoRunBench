#include <cstdlib>
#include "OpenCL.h"
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


OpenCL::OpenCL(int displayOutput)
{
	VERBOSE = displayOutput;
}

OpenCL::~OpenCL()
{
	// Flush and kill the command queue...
	clFlush(command_queue[0]);
	clFinish(command_queue[0]);
	clFlush(command_queue[1]);
	clFinish(command_queue[1]);
	
	// Release each kernel in the map kernelArray
	map<string, cl_kernel>::iterator it;
	for ( it=kernelArray.begin() ; it != kernelArray.end(); it++ )
		clReleaseKernel( (*it).second );
		
	// Now the program...
	clReleaseProgram(program);
	
	// ...and finally, the queue and context.
	clReleaseCommandQueue(command_queue[0]);
	clReleaseCommandQueue(command_queue[1]);
	clReleaseContext(context);
}

size_t OpenCL::localSize()
{
	return this->lwsize;
}

cl_command_queue OpenCL::q()
{
	return this->command_queue[0];
}

void OpenCL::launch(string toLaunch)
{
	// Launch the kernel (or at least enqueue it).
	ret = clEnqueueNDRangeKernel_fusion(command_queue, 
	//ret = clEnqueueNDRangeKernel(command_queue[0], 
	                             kernelArray[toLaunch],
	                             1,
	                             NULL,
	                             &gwsize,
	                             &lwsize,
	                             0, 
	                             NULL, 
	                             NULL);
	
	if (ret != CL_SUCCESS)
	{
		printf("\nError attempting to launch %s. Error in clCreateProgramWithSource with error code %i\n\n", toLaunch.c_str(), ret);
		exit(1);
	}
}

void OpenCL::gwSize(size_t theSize)
{
	this->gwsize = theSize;
}

cl_context OpenCL::ctxt()
{
	return this->context;
}

cl_kernel OpenCL::kernel(string kernelName)
{
	return this->kernelArray[kernelName];
}

void OpenCL::createKernel(string kernelName)
{
	cl_kernel kernel = clCreateKernel(this->program, kernelName.c_str(), NULL);
	kernelArray[kernelName] = kernel;
	
	// Get the kernel work group size.
	clGetKernelWorkGroupInfo(kernelArray[kernelName], device_id[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &lwsize, NULL);
	if (lwsize == 0)
	{
		cout << "Error: clGetKernelWorkGroupInfo() returned a max work group size of zero!" << endl;
		exit(1);
	}
	
	// Local work size must divide evenly into global work size.
	size_t howManyThreads = lwsize;
	if (lwsize > gwsize)
	{
		lwsize = gwsize;
		printf("Using %zu for local work size. \n", lwsize);
	}
	else
	{
		while (gwsize % howManyThreads != 0)
		{
			howManyThreads--;
		}
		if (VERBOSE)
			printf("Max local threads is %zu. Using %zu for local work size. \n", lwsize, howManyThreads);

		this->lwsize = howManyThreads;
	}
}

void OpenCL::buildKernel()
{
	/* Load the source code for all of the kernels into the array source_str */
	FILE*  theFile;
	char*  source_str;
	size_t source_size;
	
	theFile = fopen("kernels.cl", "r");
	if (!theFile)
	{
		fprintf(stderr, "Failed to load kernel file.\n");
		exit(1);
	}
	// Obtain length of source file.
	fseek(theFile, 0, SEEK_END);
	source_size = ftell(theFile);
	rewind(theFile);
	// Read in the file.
	source_str = (char*) malloc(sizeof(char) * (source_size + 1));
	fread(source_str, 1, source_size, theFile);
	fclose(theFile);
	source_str[source_size] = '\0';

	// Create a program from the kernel source.
	program = clCreateProgramWithSource(context,
	                                    1,
	                                    (const char **) &source_str,
	                                    NULL,           // Number of chars in kernel src. NULL means src is null-terminated.
	                                    &ret);          // Return status message in the ret variable.

	if (ret != CL_SUCCESS)
	{
		printf("\nError at clCreateProgramWithSource! Error code %i\n\n", ret);
		exit(1);
	}

	// Memory cleanup for the variable used to hold the kernel source.
	free(source_str);
	
	// Build (compile) the program.
	ret = clBuildProgram(program, NULL, NULL, NULL, NULL, NULL);
	
	if (ret != CL_SUCCESS)
	{
		printf("\nError at clBuildProgram! Error code %i\n\n", ret);
		cout << "\n*************************************************" << endl;
		cout << "***   OUTPUT FROM COMPILING THE KERNEL FILE   ***" << endl;
		cout << "*************************************************" << endl;
		// Shows the log
		char*  build_log;
		size_t log_size;
		// First call to know the proper size
		clGetProgramBuildInfo(program, NULL, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		//clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		build_log = new char[log_size + 1];
		// Second call to get the log
		clGetProgramBuildInfo(program, NULL, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
		//clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
		build_log[log_size] = '\0';
		cout << build_log << endl;
		delete[] build_log;
		cout << "\n*************************************************" << endl;
		cout << "*** END OUTPUT FROM COMPILING THE KERNEL FILE ***" << endl;
		cout << "*************************************************\n\n" << endl;
		exit(1);
	}


	/* Show error info from building the program. */
	if (VERBOSE)
	{
		cout << "\n*************************************************" << endl;
		cout << "***   OUTPUT FROM COMPILING THE KERNEL FILE   ***" << endl;
		cout << "*************************************************" << endl;
		// Shows the log
		char*  build_log;
		size_t log_size;
		// First call to know the proper size
		clGetProgramBuildInfo(program, NULL, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		//clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		build_log = new char[log_size + 1];
		// Second call to get the log
		clGetProgramBuildInfo(program, NULL, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
		//clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
		build_log[log_size] = '\0';
		cout << build_log << endl;
		delete[] build_log;
		cout << "\n*************************************************" << endl;
		cout << "*** END OUTPUT FROM COMPILING THE KERNEL FILE ***" << endl;
		cout << "*************************************************\n\n" << endl;
	}
}

void OpenCL::getDevices(cl_device_type deviceType)
{
	cl_uint         platforms_n = 0;
	cl_uint         devices_n   = 0;
	
	/* The following code queries the number of platforms and devices, and
	 * lists the information about both.
	 */
	clGetPlatformIDs(100, platform_id, &platforms_n);
	if (VERBOSE)
	{
		printf("\n=== %d OpenCL platform(s) found: ===\n", platforms_n);
		for (int i = 0; i < platforms_n; i++)
		{
			char buffer[10240];
			printf("  -- %d --\n", i);
			clGetPlatformInfo(platform_id[i], CL_PLATFORM_PROFILE, 10240, buffer,
			                  NULL);
			printf("  PROFILE = %s\n", buffer);
			clGetPlatformInfo(platform_id[i], CL_PLATFORM_VERSION, 10240, buffer,
			                  NULL);
			printf("  VERSION = %s\n", buffer);
			clGetPlatformInfo(platform_id[i], CL_PLATFORM_NAME, 10240, buffer, NULL);
			printf("  NAME = %s\n", buffer);
			clGetPlatformInfo(platform_id[i], CL_PLATFORM_VENDOR, 10240, buffer, NULL);
			printf("  VENDOR = %s\n", buffer);
			clGetPlatformInfo(platform_id[i], CL_PLATFORM_EXTENSIONS, 10240, buffer,
			                  NULL);
			printf("  EXTENSIONS = %s\n", buffer);
		}
	}


        cl_int errcode;
        cl_uint num_devices;
        char str_temp[1024];
        errcode = clGetDeviceIDs( platform_id[0], CL_DEVICE_TYPE_GPU, 1, &device_id[0], &num_devices);
        if(errcode == CL_SUCCESS) printf("number of GPU is %d\n", num_devices);
        errcode |= clGetDeviceIDs( platform_id[0], CL_DEVICE_TYPE_CPU, 1, &device_id[1], &num_devices);
        if(errcode == CL_SUCCESS) printf("number of CPU is %d\n", num_devices);
        else printf("Error getting device IDs\n");

        errcode = clGetDeviceInfo(device_id[0],CL_DEVICE_NAME, sizeof(str_temp), str_temp,NULL);
        if(errcode == CL_SUCCESS) printf("GPU device name is %s\n",str_temp);
        else printf("GPU Error getting device name\n");
        errcode = clGetDeviceInfo(device_id[1],CL_DEVICE_NAME, sizeof(str_temp), str_temp,NULL);
        if(errcode == CL_SUCCESS) printf("CPU device name is %s\n",str_temp);
        else printf("CPU Error getting device name\n");

/*	
	clGetDeviceIDs(platform_id[0], deviceType, 100, device_id, &devices_n);
	if (VERBOSE)
	{
		printf("Using the default platform (platform 0)...\n\n");
		printf("=== %d OpenCL device(s) found on platform:\n", devices_n);
		for (int i = 0; i < devices_n; i++)
		{
			char buffer[10240];
			cl_uint buf_uint;
			cl_ulong buf_ulong;
			printf("  -- %d --\n", i);
			clGetDeviceInfo(device_id[i], CL_DEVICE_NAME, sizeof(buffer), buffer,
			                NULL);
			printf("  DEVICE_NAME = %s\n", buffer);
			clGetDeviceInfo(device_id[i], CL_DEVICE_VENDOR, sizeof(buffer), buffer,
			                NULL);
			printf("  DEVICE_VENDOR = %s\n", buffer);
			clGetDeviceInfo(device_id[i], CL_DEVICE_VERSION, sizeof(buffer), buffer,
			                NULL);
			printf("  DEVICE_VERSION = %s\n", buffer);
			clGetDeviceInfo(device_id[i], CL_DRIVER_VERSION, sizeof(buffer), buffer,
			                NULL);
			printf("  DRIVER_VERSION = %s\n", buffer);
			clGetDeviceInfo(device_id[i], CL_DEVICE_MAX_COMPUTE_UNITS,
			                sizeof(buf_uint), &buf_uint, NULL);
			printf("  DEVICE_MAX_COMPUTE_UNITS = %u\n", (unsigned int) buf_uint);
			clGetDeviceInfo(device_id[i], CL_DEVICE_MAX_CLOCK_FREQUENCY,
			                sizeof(buf_uint), &buf_uint, NULL);
			printf("  DEVICE_MAX_CLOCK_FREQUENCY = %u\n", (unsigned int) buf_uint);
			clGetDeviceInfo(device_id[i], CL_DEVICE_GLOBAL_MEM_SIZE,
			                sizeof(buf_ulong), &buf_ulong, NULL);
			printf("  DEVICE_GLOBAL_MEM_SIZE = %llu\n",
			       (unsigned long long) buf_ulong);
			clGetDeviceInfo(device_id[i], CL_DEVICE_LOCAL_MEM_SIZE,
			                sizeof(buf_ulong), &buf_ulong, NULL);
			printf("  CL_DEVICE_LOCAL_MEM_SIZE = %llu\n",
			       (unsigned long long) buf_ulong);
		}
		printf("\n");
	}
*/
	
	// Create an OpenCL context.
	context = clCreateContext(NULL, 2, device_id, NULL, NULL, &ret);
	//context = clCreateContext(NULL, devices_n, device_id, NULL, NULL, &ret);
	if (ret != CL_SUCCESS)
	{
		printf("\nError at clCreateContext! Error code %i\n\n", ret);
		exit(1);
	}
 
	// Create a command queue.
	command_queue[0] = clCreateCommandQueue(context, device_id[0], 0, &ret);
	command_queue[1] = clCreateCommandQueue(context, device_id[1], 0, &ret);
	if (ret != CL_SUCCESS)
	{
		printf("\nError at clCreateCommandQueue! Error code %i\n\n", ret);
		exit(1);
	}
}

void OpenCL::init(int isGPU)
{
	if (isGPU)
		getDevices(CL_DEVICE_TYPE_GPU);
	else
		getDevices(CL_DEVICE_TYPE_CPU);

	buildKernel();
}
