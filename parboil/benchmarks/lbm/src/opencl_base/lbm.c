/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/*############################################################################*/

// includes, system
#include <CL/cl.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>

// includes, project
#include "layout_config.h"
#include "lbm_macros.h"
#include "ocl.h"
#include "lbm.h"

extern int cpu_offset ;

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
 size_t global_offset[3];
 size_t global_offset_start[3];
 size_t remain_global_work_size[3];
 int i;
 cl_int errcode;


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
//  printf("global_work_size[0]=%d, global_offset[0]=%d\n",global_work_size[0],global_offset[0]);
  remain_global_work_size[0] = global_work_size[0]-global_offset[0];
  if(remain_global_work_size[0] == 0)
    gpu_run = 0;
  if(global_offset[0] == 0)
    cpu_run = 0;

  global_offset_start[0]=global_offset[0];
  global_offset_start[1]=0;
  global_offset_start[2]=0;
 
  if(gpu_run){
    errcode = clEnqueueNDRangeKernel(command_queue[0], kernel, work_dim, global_offset_start, remain_global_work_size, local_work_size, 0, NULL, &(eventList[0]));
    if(errcode != CL_SUCCESS) printf("Error in gpu clEnqueueNDRangeKernel\n");
  }
//  clFinish(command_queue[0]);
  if(cpu_run){
//    printf("global=%d workgroup=%d \n", global_offset[0],local_work_size[0]);
    errcode = clEnqueueNDRangeKernel(command_queue[1], kernel, work_dim, NULL, global_offset, local_work_size, 0, NULL, &(eventList[1]));
    if(errcode != CL_SUCCESS) printf("Error in cpu clEnqueueNDRangeKernel %d\n",errcode );
  }
  
//  printf("global[0]=%d sum=%d, remain_global_work_size[0]=%d, global_offset[0]=%d, local_work_size=%d\n",global_work_size[0],remain_global_work_size[0]+global_offset[0],remain_global_work_size[0],global_offset[0],local_work_size[0]);
  
  if(gpu_run) errcode = clFlush(command_queue[0]);
  if(cpu_run) errcode = clFlush(command_queue[1]);
  if(gpu_run) errcode = clWaitForEvents(1,&eventList[0]);
  if(cpu_run) errcode = clWaitForEvents(1,&eventList[1]);

  return errcode;
}


/******************************************************************************/

void OpenCL_LBM_performStreamCollide( const OpenCL_Param* prm, cl_mem srcGrid, cl_mem dstGrid ) {
	 
	cl_int clStatus;

	clStatus = clSetKernelArg(prm->clKernel,0,sizeof(cl_mem),(void*)&srcGrid);
	CHECK_ERROR("clSetKernelArg")

	clStatus = clSetKernelArg(prm->clKernel,1,sizeof(cl_mem),(void*)&dstGrid);
	CHECK_ERROR("clSetKernelArg")

	size_t dimBlock[3] = {SIZE_X,1,1};
	size_t dimGrid[3] = {SIZE_X*SIZE_Y,SIZE_Z,1};
        //printf("b1=%d, b2=%d\n",SIZE_Y,SIZE_Z);
	clStatus = clEnqueueNDRangeKernel_fusion(prm->clCommandQueue,prm->clKernel,3,NULL,dimGrid,dimBlock,0,NULL,NULL); 
	//clStatus = clEnqueueNDRangeKernel(prm->clCommandQueue[0],prm->clKernel,3,NULL,dimGrid,dimBlock,0,NULL,NULL); 
	CHECK_ERROR("clEnqueueNDRangeKernel") 	
	
	clStatus = clFinish(prm->clCommandQueue[0]);
	CHECK_ERROR("clFinish")
}
/*############################################################################*/

void LBM_allocateGrid( float** ptr ) {
	const size_t size   = TOTAL_PADDED_CELLS*N_CELL_ENTRIES*sizeof( float );

	*ptr = (float*)malloc( size );
	if( ! *ptr ) {
		printf( "LBM_allocateGrid: could not allocate %.1f MByte\n",
				size / (1024.0*1024.0) );
		exit( 1 );
	}

	memset( *ptr, 0, size );

	printf( "LBM_allocateGrid: allocated %.1f MByte\n",
			size / (1024.0*1024.0) );
	
	*ptr += MARGIN;
}

/******************************************************************************/

void OpenCL_LBM_allocateGrid( const OpenCL_Param* prm, cl_mem* ptr ) {
	const size_t size = TOTAL_PADDED_CELLS*N_CELL_ENTRIES*sizeof( float );
	cl_int clStatus;
  size_t max_alloc_size = 0;
	clGetDeviceInfo(prm->clDevice[0], CL_DEVICE_MAX_MEM_ALLOC_SIZE, 
                  sizeof(max_alloc_size), &max_alloc_size, NULL);
  if (max_alloc_size < size) {
    fprintf(stderr, "Can't allocate buffer: max alloc size is %dMB\n",
            (int) (max_alloc_size >> 20));
    exit(-1);
  }
	*ptr = clCreateBuffer(prm->clContext,CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR,size,NULL,&clStatus);
	CHECK_ERROR("clCreateBuffer")
}

/*############################################################################*/

void LBM_freeGrid( float** ptr ) {
	free( *ptr-MARGIN );
	*ptr = NULL;
}

/******************************************************************************/

void OpenCL_LBM_freeGrid(cl_mem ptr) {
	clReleaseMemObject(ptr);
}

/*############################################################################*/

void LBM_initializeGrid( LBM_Grid grid ) {
	SWEEP_VAR

	SWEEP_START( 0, 0, 0, 0, 0, SIZE_Z )
	SRC_C( grid  ) = DFL1;
	SRC_N( grid  ) = DFL2;
	SRC_S( grid  ) = DFL2;
	SRC_E( grid  ) = DFL2;
	SRC_W( grid  ) = DFL2;
	SRC_T( grid  ) = DFL2;
	SRC_B( grid  ) = DFL2;
	SRC_NE( grid ) = DFL3;
	SRC_NW( grid ) = DFL3;
	SRC_SE( grid ) = DFL3;
	SRC_SW( grid ) = DFL3;
	SRC_NT( grid ) = DFL3;
	SRC_NB( grid ) = DFL3;
	SRC_ST( grid ) = DFL3;
	SRC_SB( grid ) = DFL3;
	SRC_ET( grid ) = DFL3;
	SRC_EB( grid ) = DFL3;
	SRC_WT( grid ) = DFL3;
	SRC_WB( grid ) = DFL3;
	
	CLEAR_ALL_FLAGS_SWEEP( grid );
	SWEEP_END
}

/******************************************************************************/

void OpenCL_LBM_initializeGrid( const OpenCL_Param* prm, cl_mem d_grid, LBM_Grid h_grid ) {
	const size_t size = TOTAL_PADDED_CELLS*N_CELL_ENTRIES*sizeof( float ); 
	cl_int clStatus;
	clStatus = clEnqueueWriteBuffer(prm->clCommandQueue[0],d_grid,CL_TRUE,0,size,h_grid-MARGIN,0,NULL,NULL);
	CHECK_ERROR("clEnqueueWriteBuffer")
}

void OpenCL_LBM_getDeviceGrid( const OpenCL_Param* prm, cl_mem d_grid, LBM_Grid h_grid ) {
	const size_t size = TOTAL_PADDED_CELLS*N_CELL_ENTRIES*sizeof( float );
        cl_int clStatus;
        clStatus = clEnqueueReadBuffer(prm->clCommandQueue[0],d_grid,CL_TRUE,0,size,h_grid-MARGIN,0,NULL,NULL);
	CHECK_ERROR("clEnqueueReadBuffer")
}

/*############################################################################*/

void LBM_swapGrids( cl_mem* grid1, cl_mem* grid2 ) {
	cl_mem aux = *grid1;
	*grid1 = *grid2;
	*grid2 = aux;
}

/*############################################################################*/

void LBM_loadObstacleFile( LBM_Grid grid, const char* filename ) {
	int x,  y,  z;

	FILE* file = fopen( filename, "rb" );

	for( z = 0; z < SIZE_Z; z++ ) {
		for( y = 0; y < SIZE_Y; y++ ) {
			for( x = 0; x < SIZE_X; x++ ) {
				if( fgetc( file ) != '.' ) SET_FLAG( grid, x, y, z, OBSTACLE );
			}
			fgetc( file );
		}
		fgetc( file );
	}

	fclose( file );
}

/*############################################################################*/

void LBM_initializeSpecialCellsForLDC( LBM_Grid grid ) {
	int x,  y,  z;

	for( z = -2; z < SIZE_Z+2; z++ ) {
		for( y = 0; y < SIZE_Y; y++ ) {
			for( x = 0; x < SIZE_X; x++ ) {
				if( x == 0 || x == SIZE_X-1 ||
						y == 0 || y == SIZE_Y-1 ||
						z == 0 || z == SIZE_Z-1 ) {
					SET_FLAG( grid, x, y, z, OBSTACLE );
				}
				else {
					if( (z == 1 || z == SIZE_Z-2) &&
							x > 1 && x < SIZE_X-2 &&
							y > 1 && y < SIZE_Y-2 ) {
						SET_FLAG( grid, x, y, z, ACCEL );
					}
				}
			}
		}
	}
}

/*############################################################################*/

void LBM_showGridStatistics( LBM_Grid grid ) {
	int nObstacleCells = 0,
	    nAccelCells    = 0,
	    nFluidCells    = 0;
	float ux, uy, uz;
	float minU2  = 1e+30, maxU2  = -1e+30, u2;
	float minRho = 1e+30, maxRho = -1e+30, rho;
	float mass = 0;

	SWEEP_VAR

		SWEEP_START( 0, 0, 0, 0, 0, SIZE_Z )
		rho = LOCAL( grid, C  ) + LOCAL( grid, N  )
		+ LOCAL( grid, S  ) + LOCAL( grid, E  )
		+ LOCAL( grid, W  ) + LOCAL( grid, T  )
		+ LOCAL( grid, B  ) + LOCAL( grid, NE )
		+ LOCAL( grid, NW ) + LOCAL( grid, SE )
		+ LOCAL( grid, SW ) + LOCAL( grid, NT )
		+ LOCAL( grid, NB ) + LOCAL( grid, ST )
		+ LOCAL( grid, SB ) + LOCAL( grid, ET )
		+ LOCAL( grid, EB ) + LOCAL( grid, WT )
		+ LOCAL( grid, WB );

	if( rho < minRho ) minRho = rho;
	if( rho > maxRho ) maxRho = rho;
	mass += rho;

	if( TEST_FLAG_SWEEP( grid, OBSTACLE )) {
		nObstacleCells++;
	}
	else {
		if( TEST_FLAG_SWEEP( grid, ACCEL ))
			nAccelCells++;
		else
			nFluidCells++;

		ux = + LOCAL( grid, E  ) - LOCAL( grid, W  )
			+ LOCAL( grid, NE ) - LOCAL( grid, NW )
			+ LOCAL( grid, SE ) - LOCAL( grid, SW )
			+ LOCAL( grid, ET ) + LOCAL( grid, EB )
			- LOCAL( grid, WT ) - LOCAL( grid, WB );
		uy = + LOCAL( grid, N  ) - LOCAL( grid, S  )
			+ LOCAL( grid, NE ) + LOCAL( grid, NW )
			- LOCAL( grid, SE ) - LOCAL( grid, SW )
			+ LOCAL( grid, NT ) + LOCAL( grid, NB )
			- LOCAL( grid, ST ) - LOCAL( grid, SB );
		uz = + LOCAL( grid, T  ) - LOCAL( grid, B  )
			+ LOCAL( grid, NT ) - LOCAL( grid, NB )
			+ LOCAL( grid, ST ) - LOCAL( grid, SB )
			+ LOCAL( grid, ET ) - LOCAL( grid, EB )
			+ LOCAL( grid, WT ) - LOCAL( grid, WB );
		u2 = (ux*ux + uy*uy + uz*uz) / (rho*rho);
		if( u2 < minU2 ) minU2 = u2;
		if( u2 > maxU2 ) maxU2 = u2;
	}
	SWEEP_END

		printf( "LBM_showGridStatistics:\n"
				"\tnObstacleCells: %7i nAccelCells: %7i nFluidCells: %7i\n"
				"\tminRho: %8.4f maxRho: %8.4f mass: %e\n"
				"\tminU: %e maxU: %e\n\n",
				nObstacleCells, nAccelCells, nFluidCells,
				minRho, maxRho, mass,
				sqrt( minU2 ), sqrt( maxU2 ) );

}

/*############################################################################*/

static void storeValue( FILE* file, OUTPUT_PRECISION* v ) {
	const int litteBigEndianTest = 1;
	if( (*((unsigned char*) &litteBigEndianTest)) == 0 ) {         /* big endian */
		const char* vPtr = (char*) v;
		char buffer[sizeof( OUTPUT_PRECISION )];
		int i;

		for (i = 0; i < sizeof( OUTPUT_PRECISION ); i++)
			buffer[i] = vPtr[sizeof( OUTPUT_PRECISION ) - i - 1];

		fwrite( buffer, sizeof( OUTPUT_PRECISION ), 1, file );
	}
	else {                                                     /* little endian */
		fwrite( v, sizeof( OUTPUT_PRECISION ), 1, file );
	}
}

/*############################################################################*/

static void loadValue( FILE* file, OUTPUT_PRECISION* v ) {
	const int litteBigEndianTest = 1;
	if( (*((unsigned char*) &litteBigEndianTest)) == 0 ) {         /* big endian */
		char* vPtr = (char*) v;
		char buffer[sizeof( OUTPUT_PRECISION )];
		int i;

		fread( buffer, sizeof( OUTPUT_PRECISION ), 1, file );

		for (i = 0; i < sizeof( OUTPUT_PRECISION ); i++)
			vPtr[i] = buffer[sizeof( OUTPUT_PRECISION ) - i - 1];
	}
	else {                                                     /* little endian */
		fread( v, sizeof( OUTPUT_PRECISION ), 1, file );
	}
}

/*############################################################################*/

void LBM_storeVelocityField( LBM_Grid grid, const char* filename,
		const int binary ) {
	OUTPUT_PRECISION rho, ux, uy, uz;

	FILE* file = fopen( filename, (binary ? "wb" : "w") );

	SWEEP_VAR
	SWEEP_START(0,0,0,SIZE_X,SIZE_Y,SIZE_Z)
				rho = + SRC_C( grid ) + SRC_N( grid )
					+ SRC_S( grid ) + SRC_E( grid )
					+ SRC_W( grid ) + SRC_T( grid )
					+ SRC_B( grid ) + SRC_NE( grid )
					+ SRC_NW( grid ) + SRC_SE( grid )
					+ SRC_SW( grid ) + SRC_NT( grid )
					+ SRC_NB( grid ) + SRC_ST( grid )
					+ SRC_SB( grid ) + SRC_ET( grid )
					+ SRC_EB( grid ) + SRC_WT( grid )
					+ SRC_WB( grid );
				ux = + SRC_E( grid ) - SRC_W( grid ) 
					+ SRC_NE( grid ) - SRC_NW( grid ) 
					+ SRC_SE( grid ) - SRC_SW( grid ) 
					+ SRC_ET( grid ) + SRC_EB( grid ) 
					- SRC_WT( grid ) - SRC_WB( grid );
				uy = + SRC_N( grid ) - SRC_S( grid ) 
					+ SRC_NE( grid ) + SRC_NW( grid ) 
					- SRC_SE( grid ) - SRC_SW( grid ) 
					+ SRC_NT( grid ) + SRC_NB( grid ) 
					- SRC_ST( grid ) - SRC_SB( grid );
				uz = + SRC_T( grid ) - SRC_B( grid ) 
					+ SRC_NT( grid ) - SRC_NB( grid ) 
					+ SRC_ST( grid ) - SRC_SB( grid ) 
					+ SRC_ET( grid ) - SRC_EB( grid ) 
					+ SRC_WT( grid ) - SRC_WB( grid );
				ux /= rho;
				uy /= rho;
				uz /= rho;

				if( binary ) {
					/*
					   fwrite( &ux, sizeof( ux ), 1, file );
					   fwrite( &uy, sizeof( uy ), 1, file );
					   fwrite( &uz, sizeof( uz ), 1, file );
					   */
					storeValue( file, &ux );
					storeValue( file, &uy );
					storeValue( file, &uz );
				} else
					fprintf( file, "%e %e %e\n", ux, uy, uz );

	SWEEP_END;

	fclose( file );
}
