include ../common/make.config
CC = gcc
CC_FLAGS = -std=c99

override VERIFY = -DVERIFY
override OUTPUT = -DOUTPUT
override TIMER = -DTIMER


ifdef VERIFY
override VERIFY = -DVERIFY
endif

ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

ifdef TIMER
override TIMER = -DTIMER
endif

hybridsort: hybridsort.c bucketsort.h mergesort.h bucketsort.c mergesort.c fusion.c
	$(CC) $(CC_FLAGS) -o hybridsort $(VERIFY) $(OUTPUT) $(TIMER) -I$(OPENCL_INC) $(HISTO_WG_SIZE_0) $(BUCKET_WG_SIZE_0) $(BUCKET_WG_SIZE_1) $(MERGE_WG_SIZE_0) $(MERGE_WG_SIZE_1) fusion.c hybridsort.c bucketsort.c mergesort.c -L$(OPENCL_LIB) -lOpenCL -lm 


clean:
	rm -f  hybridsort
