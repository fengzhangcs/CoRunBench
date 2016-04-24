// #ifdef __cplusplus
// extern "C" {
// #endif

//========================================================================================================================================================================================================200
//	DEFINE/INCLUDE
//========================================================================================================================================================================================================200

#include "define.h"

//======================================================================================================================================================150
//	DEFINE
//======================================================================================================================================================150

// double precision support (switch between as needed for NVIDIA/AMD)
//#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_amd_fp64 : enable

// clBuildProgram compiler cannot link this file for some reason, so had to redefine constants and structures below
// #include ../common.h						// (in directory specified to compiler)			main function header

//======================================================================================================================================================150
//	DEFINE (had to bring from ../common.h here because feature of including headers in clBuildProgram does not work for some reason)
//======================================================================================================================================================150

// change to double if double precision needed
// #define fp float

//======================================================================================================================================================150
//	STRUCTURES (had to bring from ../common.h here because feature of including headers in clBuildProgram does not work for some reason)
//======================================================================================================================================================150

// Type representing the record to which a given key refers. In a real B+ tree system, the record would hold data (in a database) or a file (in an operating system) or some other information.
// Users can rewrite this part of the code to change the type and content of the value field.
typedef struct record {
	int value;
} record;

// ???
typedef struct knode {
	int location;
	int indices [DEFAULT_ORDER + 1];
	int  keys [DEFAULT_ORDER + 1];
	bool is_leaf;
	int num_keys;
} knode; 

//========================================================================================================================================================================================================200
//	findK function
//========================================================================================================================================================================================================200

__kernel void 
findK(	long height,
		__global knode *knodesD,
		long knodes_elem,
		__global record *recordsD,

		__global long *currKnodeD,
		__global long *offsetD,
		__global int *keysD, 
		__global record *ansD,
		 int groupOffset)
{

	// private thread IDs
	int thid = get_local_id(0);
	//int groupOffset=get_global_offset(0)/get_local_size(0);	
	int bid = get_group_id(0)+groupOffset;
	int new_thid = thid;

	// make few passes since there are more elements than threads in a block
	while(new_thid < DEFAULT_ORDER){

		// process tree levels
		int i;
		for(i = 0; i < height; i++){

			// if value is between the two keys
			if((knodesD[currKnodeD[bid]].keys[new_thid]) <= keysD[bid] && (knodesD[currKnodeD[bid]].keys[new_thid+1] > keysD[bid])){
				// this conditional statement is inserted to avoid crush due to but in original code
				// "offset[bid]" calculated below that addresses knodes[] in the next iteration goes outside of its bounds cause segmentation fault
				// more specifically, values saved into knodes->indices in the main function are out of bounds of knodes that they address
				if(knodesD[offsetD[bid]].indices[new_thid] < knodes_elem){
					offsetD[bid] = knodesD[offsetD[bid]].indices[new_thid];
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

			// set for next tree level
			if(new_thid==0){
				currKnodeD[bid] = offsetD[bid];
			}
			barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		}

		//At this point, we have a candidate leaf node which may contain
		//the target record.  Check each key to hopefully find the record
		if(knodesD[currKnodeD[bid]].keys[new_thid] == keysD[bid]){
			ansD[bid].value = recordsD[knodesD[currKnodeD[bid]].indices[new_thid]].value;
		}
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		// go for another round
		new_thid = new_thid + THREADS;

	}

}

//========================================================================================================================================================================================================200
//	End
//========================================================================================================================================================================================================200

// #ifdef __cplusplus
// }
// #endif
