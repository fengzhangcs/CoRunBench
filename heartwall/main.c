//========================================================================================================================================================================================================200
//======================================================================================================================================================150
//====================================================================================================100
//==================================================50

//========================================================================================================================================================================================================200
//	DEFINE / INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	LIBRARIES
//======================================================================================================================================================150

#include <stdlib.h>								// (in directory known to compiler)
#include <math.h>								// (in directory known to compiler)
#include <string.h>								// (in directory known to compiler)

//======================================================================================================================================================150
//	MAIN FUNCTION HEADER
//======================================================================================================================================================150

#include "./main.h"								// (in the main program folder)	needed to recognized input parameters

//======================================================================================================================================================150
//	UTILITIES
//======================================================================================================================================================150

#include "./util/avi/avilib.h"					// (in directory)							needed by avi functions
#include "./util/avi/avimod.h"					// (in directory)							needed by avi functions

//======================================================================================================================================================150
//	KERNEL
//======================================================================================================================================================150

#include "./kernel/kernel_gpu_opencl_wrapper.h"

//======================================================================================================================================================150
//	END
//======================================================================================================================================================150
#define mark
int cpu_offset;
int tony_device;
params_common common;
int i;
avi_t* frames;
// variables
char* video_file_name;
int* endoRow;
int* endoCol;
int* tEndoRowLoc;
int* tEndoColLoc;
int* epiRow;
int* epiCol;
int* tEpiRowLoc;
int* tEpiColLoc;

void Inputread(){

	//======================================================================================================================================================150
	//	INPUTS
	//======================================================================================================================================================150

	//====================================================================================================100
	//	READ PARAMETERS FROM FILE
	//====================================================================================================100

	read_parameters(	"./input/input.txt",
						&common.tSize,
						&common.sSize,
						&common.maxMove,
						&common.alpha);

	//====================================================================================================100
	//	READ SIZE OF INPUTS FROM FILE
	//====================================================================================================100

	read_header(	"./input/input.txt",
					&common.endoPoints,
					&common.epiPoints);

	common.allPoints = common.endoPoints + common.epiPoints;

	//====================================================================================================100
	//	READ DATA FROM FILE
	//====================================================================================================100

	//==================================================50
	//	ENDO POINTS MEMORY ALLOCATION
	//==================================================50

	common.endo_mem = sizeof(int) * common.endoPoints;

	endoRow = (int*)malloc(common.endo_mem);
	endoCol = (int*)malloc(common.endo_mem);
	tEndoRowLoc = (int*)malloc(common.endo_mem * common.no_frames);
	tEndoColLoc = (int*)malloc(common.endo_mem * common.no_frames);

	//==================================================50
	//	EPI POINTS MEMORY ALLOCATION
	//==================================================50

	common.epi_mem = sizeof(int) * common.epiPoints;

	epiRow = (int *)malloc(common.epi_mem);
	epiCol = (int *)malloc(common.epi_mem);
	tEpiRowLoc = (int *)malloc(common.epi_mem * common.no_frames);
	tEpiColLoc = (int *)malloc(common.epi_mem * common.no_frames);

	//==================================================50
	//	READ DATA FROM FILE
	//==================================================50

	read_data(	"./input/input.txt",
				common.endoPoints,
				endoRow,
				endoCol,
				common.epiPoints,
				epiRow,
				epiCol);

	//==================================================50
	//	End
	//==================================================50

	//====================================================================================================100
	//	End
	//====================================================================================================100


	//======================================================================================================================================================150
	//	KERNELL WRAPPER CALL
	//======================================================================================================================================================150



}
float command(){
	kernel_gpu_opencl_wrapper(	common,
								endoRow,
								endoCol,
								tEndoRowLoc,
								tEndoColLoc,
								epiRow,
								epiCol,
								tEpiRowLoc,
								tEpiColLoc,
								frames);



	//======================================================================================================================================================150
	//	DEALLOCATION
	//======================================================================================================================================================150

	//====================================================================================================100
	// endo points
	//====================================================================================================100
//========================================================================================================================================================================================================200
}
//========================================================================================================================================================================================================200
//	MAIN FUNCTION
//========================================================================================================================================================================================================200

void write_data(	char* filename,
			int frameNo,
			int frames_processed,
			int endoPoints,
			int* input_a,
			int* input_b,
			int epiPoints,
			int* input_2a,
			int* input_2b){

	//================================================================================80
	//	VARIABLES
	//================================================================================80

	FILE* fid;
	int i,j;
	char c;

	//================================================================================80
	//	OPEN FILE FOR READING
	//================================================================================80

	fid = fopen(filename, "w+");
	if( fid == NULL ){
		printf( "The file was not opened for writing\n" );
		return;
	}


	//================================================================================80
	//	WRITE VALUES TO THE FILE
	//================================================================================80
      fprintf(fid, "Total AVI Frames: %d\n", frameNo);	
      fprintf(fid, "Frames Processed: %d\n", frames_processed);	
      fprintf(fid, "endoPoints: %d\n", endoPoints);
      fprintf(fid, "epiPoints: %d", epiPoints);
	for(j=0; j<frames_processed;j++)
	  {
	    fprintf(fid, "\n---Frame %d---",j);
	    fprintf(fid, "\n--endo--\n",j);
	    for(i=0; i<endoPoints; i++){
	      fprintf(fid, "%d\t", input_a[j+i*frameNo]);
	    }
	    fprintf(fid, "\n");
	    for(i=0; i<endoPoints; i++){
	      // if(input_b[j*size+i] > 2000) input_b[j*size+i]=0;
	      fprintf(fid, "%d\t", input_b[j+i*frameNo]);
	    }
	    fprintf(fid, "\n--epi--\n",j);
	    for(i=0; i<epiPoints; i++){
	      //if(input_2a[j*size_2+i] > 2000) input_2a[j*size_2+i]=0;
	      fprintf(fid, "%d\t", input_2a[j+i*frameNo]);
	    }
	    fprintf(fid, "\n");
	    for(i=0; i<epiPoints; i++){
	      //if(input_2b[j*size_2+i] > 2000) input_2b[j*size_2+i]=0;
	      fprintf(fid, "%d\t", input_2b[j+i*frameNo]);
	    }
	  }
	// 	================================================================================80
	//		CLOSE FILE
		  //	================================================================================80

	fclose(fid);

}


int 
main(	int argc, 
		char* argv []){

	//======================================================================================================================================================150
	//	VARIABLES
	//======================================================================================================================================================150



	//======================================================================================================================================================150
	//	STRUCTURES, GLOBAL STRUCTURE VARIABLES
	//======================================================================================================================================================150


	common.common_mem = sizeof(params_common);

	//======================================================================================================================================================150
	// 	FRAME INFO
	//======================================================================================================================================================150



	// open movie file
 	video_file_name = (char *) "./input/input.avi";
	frames = (avi_t*)AVI_open_input_file(video_file_name, 1);														// added casting
	if (frames == NULL)  {
		   AVI_print_error((char *) "Error with AVI_open_input_file");
		   return -1;
	}

	// dimensions
	common.no_frames = AVI_video_frames(frames);
	common.frame_rows = AVI_video_height(frames);
	common.frame_cols = AVI_video_width(frames);
	common.frame_elem = common.frame_rows * common.frame_cols;
	common.frame_mem = sizeof(fp) * common.frame_elem;



	//======================================================================================================================================================150
	// 	CHECK INPUT ARGUMENTS
	//======================================================================================================================================================150
	int cur_arg;
	for(cur_arg=1; cur_arg<argc; cur_arg++){
		if(strcmp(argv[cur_arg], "-no_frames")==0){
			// check if value provided
			if(argc>=cur_arg+1){
				common.frames_processed = atoi(argv[cur_arg+1]);
				if(common.frames_processed<0 || common.frames_processed>common.no_frames){
					printf("ERROR: %d is an incorrect number of frames specified, select in the range of 0-%d\n", common.frames_processed, common.no_frames);
					return 0;
				}
				cur_arg = cur_arg+1;
			}
			// value not provided
			else{
				printf("ERROR: missing argument (number of frames to processed)\n");
				return 0;
			}
		}else if(strcmp(argv[cur_arg], "-cpu_offset")==0){
			
			// check if value provided
			if(argc>=cur_arg+1){
				cpu_offset = atoi(argv[cur_arg+1]);
				//printf("cpu_offset %d \n",cpu_offset);
				cur_arg = cur_arg+1;
				// value is not a number
			}
			else{
				printf("ERROR: Missing value to -offset parameter\n");
				return 0;
			}
		}
		else if(strcmp(argv[cur_arg], "-device")==0){
			
			// check if value provided
			if(argc>=cur_arg+1){
				tony_device = atoi(argv[cur_arg+1]);
				//printf("cpu_offset %d \n",cpu_offset);
				cur_arg = cur_arg+1;
				// value is not a number
			}
			else{
				printf("ERROR: Missing value to -offset parameter\n");
				return 0;
			}
		}
	}
	Inputread();
	command();
//#define OUTPUT
#ifdef OUTPUT
        //zf
	write_data(	"result.txt",
			common.no_frames,
			common.frames_processed,		
				common.endoPoints,
				tEndoRowLoc,
				tEndoColLoc,
				common.epiPoints,
				tEpiRowLoc,
				tEpiColLoc);

#endif



#ifdef mark
	float totaltime=0;
	totaltime+=command();
	//totaltime+=command();
	//totaltime+=command();
	//printf("CAUTION: command, offset %d\t total time %.2f ms\n",cpu_offset,totaltime*1000);
//	printf("CAUTION: command, offset %d\t total time %.2f ms\n",cpu_offset,totaltime/3*1000);
	printf("CAUTION: cpu_offset: %d time: %f mseconds\n",cpu_offset,totaltime/1*1000);
#endif

	free(endoRow);
	free(endoCol);
	free(tEndoRowLoc);
	free(tEndoColLoc);

	//====================================================================================================100
	// epi points
	//====================================================================================================100

	free(epiRow);
	free(epiCol);
	free(tEpiRowLoc);
	free(tEpiColLoc);

	//====================================================================================================100
	//	End
	//====================================================================================================100



//========================================================================================================================================================================================================200
//	End
	return 0;
}
