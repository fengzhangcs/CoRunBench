#include <stdio.h>
#include <stdlib.h>
#include <string.h>
FILE *ifp, *ofp;
char *mode = "r";
char *inputFilename;
long size;
int main(int argc ,char** argv){
	int cur_arg;
	for(cur_arg=1;cur_arg<argc;cur_arg++){
		if(strcmp(argv[cur_arg],"-file")==0){
			if(argc >=cur_arg+1){
				inputFilename=argv[++cur_arg];	
				printf("inputFileName is %s \n",inputFilename);
			}
			else{
				printf("ERROR: must specified file name\n");
				return -1;
			}
		}else if(strcmp(argv[cur_arg],"-size")==0){
			if(argc >=cur_arg+1){
				size=atoi(argv[++cur_arg]);
				printf("inputsize is %d \n",size);
			}
			else{
				printf("ERROR: must specified input size\n");
				return -1;
			}
		}
	}
	ofp = fopen(inputFilename, "w");
	if (ofp == NULL) {
  		fprintf(stderr, "Can't open output file %s!\n",
       	   inputFilename);
  		exit(1);
	}
	long i;
	fprintf(ofp,"%d\n",size);
	for(i=0;i<size;i++){
		fprintf(ofp,"%d\n",i);	
	}
}
