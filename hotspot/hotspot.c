#include "hotspot.h"
#include "tony.c"
void writeoutput(float *vect, int grid_rows, int grid_cols, char *file) {

	int i,j, index=0;
	FILE *fp;
	char str[STR_SIZE];

	if( (fp = fopen(file, "w" )) == 0 )
          printf( "The file was not opened\n" );


	for (i=0; i < grid_rows; i++) 
	 for (j=0; j < grid_cols; j++)
	 {

		 sprintf(str, "%d\t%g\n", index, vect[i*grid_cols+j]);
		 fputs(str,fp);
		 index++;
	 }
		
      fclose(fp);	
}


void readinput(float *vect, int grid_rows, int grid_cols, char *file) {

  	int i,j;
	FILE *fp;
	char str[STR_SIZE];
	float val;

	if( (fp  = fopen(file, "r" )) ==0 )
            fatal( "The file was not opened" );


	for (i=0; i <= grid_rows-1; i++) 
	 for (j=0; j <= grid_cols-1; j++)
	 {
		if (fgets(str, STR_SIZE, fp) == NULL) fatal("Error reading file\n");
		if (feof(fp))
			fatal("not enough lines in file");
		//if ((sscanf(str, "%d%f", &index, &val) != 2) || (index != ((i-1)*(grid_cols-2)+j-1)))
		if ((sscanf(str, "%f", &val) != 1))
			fatal("invalid file format");
		vect[i*grid_cols+j] = val;
	}

	fclose(fp);	

}


/*
   compute N time steps
*/

int compute_tran_temp(cl_mem MatrixPower, cl_mem MatrixTemp[2], int col, int row, \
		int total_iterations, int num_iterations, int blockCols, int blockRows, int borderCols, int borderRows,
		float *TempCPU, float *PowerCPU) 
{ 
	
	float grid_height = chip_height / row;
	float grid_width = chip_width / col;

	float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
	float Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
	float Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
	float Rz = t_chip / (K_SI * grid_height * grid_width);

	float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
	float step = PRECISION / max_slope;
	int t;

	int src = 0, dst = 1;
	
	cl_int error;
	
	// Determine GPU work group grid
	size_t global_work_size[2];
	global_work_size[0] = BLOCK_SIZE * blockCols;
	//printf("global_work_size[0] %d\n",global_work_size[0]);
	global_work_size[1] = BLOCK_SIZE * blockRows;
	//printf("global_work_size[1] %d\n",global_work_size[1]);
	size_t local_work_size[2];
	local_work_size[0] = BLOCK_SIZE;
	local_work_size[1] = BLOCK_SIZE;
	
	
	long long start_time = get_time();	
	
	for (t = 0; t < total_iterations; t += num_iterations) {
		
		// Specify kernel arguments
		int iter = MIN(num_iterations, total_iterations - t);
		clSetKernelArg(kernel, 0, sizeof(int), (void *) &iter);
		clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &MatrixPower);
		clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &MatrixTemp[src]);
		clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &MatrixTemp[dst]);
		clSetKernelArg(kernel, 4, sizeof(int), (void *) &col);
		clSetKernelArg(kernel, 5, sizeof(int), (void *) &row);
		clSetKernelArg(kernel, 6, sizeof(int), (void *) &borderCols);
		clSetKernelArg(kernel, 7, sizeof(int), (void *) &borderRows);
		clSetKernelArg(kernel, 8, sizeof(float), (void *) &Cap);
		clSetKernelArg(kernel, 9, sizeof(float), (void *) &Rx);
		clSetKernelArg(kernel, 10, sizeof(float), (void *) &Ry);
		clSetKernelArg(kernel, 11, sizeof(float), (void *) &Rz);
		clSetKernelArg(kernel, 12, sizeof(float), (void *) &step);
		
		// Launch kernel
		error = clEnqueueNDRangeKernel_tony(command_queue, kernel, 2, global_work_size, local_work_size);
		if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
		
		// Swap input and output GPU matrices
		src = 1 - src;
		dst = 1 - dst;
	}
	
	long long end_time = get_time();
	long long total_time = (end_time - start_time);	
	printf("\nCAUTION: cpu_offset: %d time: %f mseconds\n",cpu_offset , ((float) total_time) / (1000));	
	//printf("\nCAUTION:Kernel time: %.3f mseconds \t cpu_offset is %d\n", ((float) total_time) / (1000),cpu_offset);	
	return src;
}

void usage(int argc, char **argv) {
	fprintf(stderr, "Usage: %s <grid_rows/grid_cols> <pyramid_height> <sim_time> <temp_file> <power_file> <output_file>\n", argv[0]);
	fprintf(stderr, "\t<grid_rows/grid_cols>  - number of rows/cols in the grid (positive integer)\n");
	fprintf(stderr, "\t<pyramid_height> - pyramid heigh(positive integer)\n");
	fprintf(stderr, "\t<sim_time>   - number of iterations\n");
	fprintf(stderr, "\t<temp_file>  - name of the file containing the initial temperature values of each cell\n");
	fprintf(stderr, "\t<power_file> - name of the file containing the dissipated power values of each cell\n");
	fprintf(stderr, "\t<output_file> - name of the output file\n");
	exit(1);
}

int main(int argc, char** argv) {


	cl_int error;
	cl_uint num_platforms;
	
	// Get the number of platforms
	error = clGetPlatformIDs(0, NULL, &num_platforms);
    if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
	
	// Get the list of platforms
	cl_platform_id* platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id) * num_platforms);
	error = clGetPlatformIDs(num_platforms, platforms, NULL);
    if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
	
	// Print the chosen platform (if there are multiple platforms, choose the first one)
	cl_platform_id platform = platforms[0];
	char pbuf[100];
	error = clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, sizeof(pbuf), pbuf, NULL);
	if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
	printf("Platform: %s\n", pbuf);
//====================================================================================================100
	//	GET DEVICES AVAILABLE FOR THE Platform, SELECT all
	//====================================================================================================100	
	cl_uint uiNumDevices;
    error = clGetDeviceIDs(platform,CL_DEVICE_TYPE_ALL,0,NULL,&uiNumDevices);
	if (error != CL_SUCCESS)
  		fatal_CL(error, __LINE__);
	 //printf("uiNumDevices :%d \n", uiNumDevices);
    device=(cl_device_id*)malloc(sizeof(cl_device_id)*2);
    cl_device_id alldevice[2];
	error = clGetDeviceIDs(platform,CL_DEVICE_TYPE_CPU,1,alldevice,NULL);

    if (error != CL_SUCCESS)
        fatal_CL(error, __LINE__);
	device[0]=alldevice[0];
    error = clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,1,alldevice,NULL);
	if (error != CL_SUCCESS)
        fatal_CL(error, __LINE__);
	device[1]=alldevice[0];
    error = clGetDeviceInfo(device[0],
					CL_DEVICE_NAME,
                                       sizeof(pbuf),
                                        pbuf,
                                        NULL);
    if (error != CL_SUCCESS)
        fatal_CL(error, __LINE__);
        //printf("Device: %s\n", pbuf);
		
	error = clGetDeviceInfo(device[1],
                                                        CL_DEVICE_NAME,
                                                        sizeof(pbuf),
                                                        pbuf,
                                                        NULL);
    if (error != CL_SUCCESS)
        fatal_CL(error, __LINE__);
	
	//====================================================================================================100
	//	CREATE CONTEXT FOR ALL devices
	//====================================================================================================100

	context = clCreateContext(0,uiNumDevices,device,NULL,NULL,&error);
	if (error != CL_SUCCESS)
                fatal_CL(error, __LINE__);
	//====================================================================================================100
	//	CREATE COMMAND QUEUE FOR THE DEVICE
	//====================================================================================================100
	command_queue=(cl_command_queue*)malloc(sizeof(cl_command_queue)*2);	
	// Create a command queue
	int CPU_GPU;
	for(CPU_GPU=0;CPU_GPU<2;CPU_GPU++){
		command_queue[CPU_GPU] = clCreateCommandQueue(	context, 
											device[CPU_GPU], 
											0, 
											&error);
		if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	}	
	
	
	

    int size;
    int grid_rows,grid_cols = 0;
    float *FilesavingTemp,*FilesavingPower; //,*MatrixOut; 
    char *tfile, *pfile, *ofile;
    
    int total_iterations = 60;
    int pyramid_height = 1; // number of iterations
	
	if (argc < 8)
		usage(argc, argv);
	if((grid_rows = atoi(argv[1]))<=0||
	   (grid_cols = atoi(argv[1]))<=0||
       (pyramid_height = atoi(argv[2]))<=0||
       (total_iterations = atoi(argv[3]))<=0)
		usage(argc, argv);
		
	tfile=argv[4];
    pfile=argv[5];
    ofile=argv[6];
    cpu_offset=atoi(argv[7]);	
    tony_device = atoi(argv[8]);
    size=grid_rows*grid_cols;

    // --------------- pyramid parameters --------------- 
    int borderCols = (pyramid_height)*EXPAND_RATE/2;
    int borderRows = (pyramid_height)*EXPAND_RATE/2;
    int smallBlockCol = BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
    int smallBlockRow = BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
    int blockCols = grid_cols/smallBlockCol+((grid_cols%smallBlockCol==0)?0:1);
	//printf("blockCols is %d\n",blockCols);
    int blockRows = grid_rows/smallBlockRow+((grid_rows%smallBlockRow==0)?0:1);
	//printf("blockRows is %d\n",blockRows);

    FilesavingTemp = (float *) malloc(size*sizeof(float));
    FilesavingPower = (float *) malloc(size*sizeof(float));
    // MatrixOut = (float *) calloc (size, sizeof(float));

    if( !FilesavingPower || !FilesavingTemp) // || !MatrixOut)
        fatal("unable to allocate memory");
	
	// Read input data from disk
    readinput(FilesavingTemp, grid_rows, grid_cols, tfile);
    readinput(FilesavingPower, grid_rows, grid_cols, pfile);
	
	// Load kernel source from file
	const char *source = load_kernel_source("hotspot_kernel.cl");
	size_t sourceSize = strlen(source);
	
	// Compile the kernel
    cl_program program = clCreateProgramWithSource(context, 1, &source, &sourceSize, &error);
    if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
	
    // Create an executable from the kernel
	error = clBuildProgram(program, 2, device, NULL, NULL, NULL);
	// Show compiler warnings/errors
	static char log[65536]; memset(log, 0, sizeof(log));
	clGetProgramBuildInfo(program, device[0], CL_PROGRAM_BUILD_LOG, sizeof(log)-1, log, NULL);
	if (strstr(log,"warning:") || strstr(log, "error:")) printf("<<<<\n%s\n>>>>\n", log);
    if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
    kernel = clCreateKernel(program, "hotspot", &error);
    if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
	
		
	long long start_time = get_time();
	
	// Create two temperature matrices and copy the temperature input data
	cl_mem MatrixTemp[2];
	// Create input memory buffers on device
	MatrixTemp[0] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * size, FilesavingTemp, &error);
	if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
	MatrixTemp[1] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * size, NULL, &error);
	if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
	
	// Copy the power input data
	cl_mem MatrixPower = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * size, FilesavingPower, &error);
	if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
	
	// Perform the computation
	int ret = compute_tran_temp(MatrixPower, MatrixTemp, grid_cols, grid_rows, total_iterations, pyramid_height,
								blockCols, blockRows, borderCols, borderRows, FilesavingTemp, FilesavingPower);
	
	// Copy final temperature data back
	cl_float *MatrixOut = (cl_float *) clEnqueueMapBuffer(command_queue[0], MatrixTemp[ret], CL_TRUE, CL_MAP_READ, 0, sizeof(float) * size, 0, NULL, NULL, &error);
	if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
	
	long long end_time = get_time();	
	printf("Total time: %.3f seconds\n", ((float) (end_time - start_time)) / (1000*1000));
	
	// Write final output to output file
    writeoutput(MatrixOut, grid_rows, grid_cols, ofile);
    
	error = clEnqueueUnmapMemObject(command_queue[0], MatrixTemp[ret], (void *) MatrixOut, 0, NULL, NULL);
	if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
	
	clReleaseMemObject(MatrixTemp[0]);
	clReleaseMemObject(MatrixTemp[1]);
	clReleaseMemObject(MatrixPower);
	
        clReleaseContext(context);

	return 0;
}
