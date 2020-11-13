#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <matrix.h>
#include "mex.h"


#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif


void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[])
{
	
	int S = 128;
	
	/* Declare kernel and other device setup */
	cl_platform_id cpPlatform;        	// OpenCL platform
	cl_device_id device_id;           	// device ID
	cl_int err;							// error command
	
	////////////////////////////////////* SETUP HARDWARE *///////////////////////////////////////////// 
	
	char* deviceInfo;
	size_t infoSize;
    cl_int MCU;
	/* Platform Setup */
	err = clGetPlatformIDs(1, &cpPlatform, NULL); 
	if(err != CL_SUCCESS ) {
		mexPrintf("Couldn't identify a platform");
		return;
	}
	
	/* ID of device */
	err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
	if(err == CL_DEVICE_NOT_FOUND) {
		/* If error setup for CPU */
		err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
	}
	
	/* Print device ID*/
	clGetDeviceInfo(device_id, CL_DEVICE_NAME, 0, NULL, &infoSize);
	deviceInfo = (char*) malloc(infoSize);
	clGetDeviceInfo(device_id, CL_DEVICE_NAME, infoSize, deviceInfo, NULL);
	mexPrintf("Computing Device: %s\n", deviceInfo);
	
	clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(MCU), &MCU, NULL);
	mexPrintf("Compute Units Available: %u\n", MCU);
	
	/* CL_DEVICE_GLOBAL_MEM_SIZE */
	cl_ulong mem_size;
	clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
	printf("  CL_DEVICE_GLOBAL_MEM_SIZE:\t\t%u MByte\n", (unsigned int)(mem_size / (1024 * 1024)));
	
	// CL_DEVICE_LOCAL_MEM_TYPE
	cl_device_local_mem_type local_mem_type;
	clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(local_mem_type), &local_mem_type, NULL);
	printf("  CL_DEVICE_LOCAL_MEM_TYPE:\t\t%s\n", local_mem_type == 1 ? "local" : "global");

	// CL_DEVICE_LOCAL_MEM_SIZE
	clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
	printf("  CL_DEVICE_LOCAL_MEM_SIZE:\t\t%u KByte\n", (unsigned int)(mem_size / 1024));

	// CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE
	clGetDeviceInfo(device_id, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(mem_size), &mem_size, NULL);
	printf("  CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE:\t%u KByte\n", (unsigned int)(mem_size / 1024));
	
	// CL_DEVICE_LOCAL_MEM_SIZE
	clGetDeviceInfo(device_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE , sizeof(mem_size), &mem_size, NULL);
	printf("  CL_DEVICE_MAX_MEM_ALLOC_SIZE :\t\t%u MByte\n", (unsigned int)(mem_size / (1024 * 1024)));
	
	/* Define local and global sizes - will depend on swarm size and cores of GPU */
	size_t localSize, globalSize;
	globalSize = S;	// Total Number of work items
	localSize = (size_t)ceil((double)globalSize/32.0);
	
	mexPrintf("Global Size = %d\n", globalSize);
	mexPrintf("Local Size = %d\n", localSize);
	
	free(deviceInfo);
	/* Check for any errors */
	if(err != CL_SUCCESS ) {
		mexPrintf("Couldn't access any devices");
		return;   
	}
	
	
	mexPrintf("CONFIG() COMPLETE. DEVICES FOUND\n\n\n ");
}