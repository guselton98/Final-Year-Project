#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <matrix.h>
#include "mex.h"

//openCL config/setup process based on https://github.com/rsnemmen/OpenCL-examples
#define SOURCE_FILES {"PPSO.cl"}

//Binary file
#define BINARY_FILE "binCache.txt"
#define NUM_FILES 1

/* Swarm SIze and kernel function definition */
#define SAMPLES 2000
#define MEMORY_TIME 70
#define ML_CONST 12*12
#define ML2_CONST 78*78

#define ML_TOT ML_CONST+ML2_CONST
#define N_SIZE SAMPLES-MEMORY_TIME+1


#define KERNEL_FUNC "PPSO"


#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

/* Files */
FILE *fileID;


/* Function definitions */
void SquareMatrixMul(double A[], double B[], double C[], int N);
int CHOL(double A[], double L[], int N);

void getErrorString(cl_int error);

/* Define Particle Properties - 2nd order only*/
typedef struct
{
	/* PSO- doesnt */
	cl_double position[6];
	cl_double pcost;
	cl_double velocity[6];
	cl_double pbest[6];
	cl_double pbcost;
}particle;

/* HARDCODED :D */
typedef struct
{
	/* Cost - ml = 12, N = 400, n = 70 */
	cl_double prior1[12*12];
	cl_double prior2[78*78];
	cl_double sig[731*731];
	cl_double Q[731*731];
}costCalc;

/* HARDCODED :D */
typedef struct
{
	cl_double phi[731*90];
	cl_double U[78];
	cl_double V[78];
	cl_double Y[731];
}dataset;

/* Data Structure */
/* y[N-n+1], u,v[(M^2+M)/2], PHI[(N-n+1)*(M+(M^2+M)/2)] */


/* Main mex function, called from MATLAB */
void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[])
{
	/* Received Data [ul, ll, term, prop, data] */
	double *u_limits, *l_limits, *termCri, *swarmProp;
    u_limits = mxGetPr(prhs[0]);
    l_limits = mxGetPr(prhs[1]);
    termCri = mxGetPr(prhs[2]);
    swarmProp = mxGetPr(prhs[3]);
	
	/* Create mex buffers */
	mxArray *N, *M, *ML, *PHI, *Y, *V, *U;
		
	/* Scalar */
	/* n = data sample size, m = time domain kernel size, ml = laguerre domain kernel size */
	int n, m, ml;
	
	/* Iterative , temprorary variables */
    int i , j, k;
	double temp2;
	
	/* Check particle displacement */
	double rmse;
	double *dimensions;
	int lock;
	double ul_dif = 0;
	dimensions = (double *)malloc(sizeof(double)*6);
	/* Termination Criteria, term = [maxIter, maxDis, maxCost]; */
	double maxIter, maxDis, maxCost;
	maxIter = termCri[0];
	maxDis = termCri[1];
	maxCost = termCri[2];
	
	/* Extract PSO Information */
	int S, d;
    S = (int)swarmProp[0];
    d = (int)swarmProp[1];
	double C1, C2, inertia;
	C1 = swarmProp[2];
    C2 = swarmProp[3];
	inertia = swarmProp[4];
	
	/* Extract Scalar sizes needed for malloc and multiplications */
	if (mxIsStruct(prhs[4]))
	{
        N = mxGetField(prhs[4], 0, "n"); 
        n = (int)mxGetScalar(N);
		M = mxGetField(prhs[4], 0, "m"); 
        m = (int)mxGetScalar(M);
		ML = mxGetField(prhs[4], 0, "ml"); 
        ml = (int)mxGetScalar(ML);
		mexPrintf("N = %d \t Time memory legnth = %d \t Laguerre Memory length = %d\n", n, m, ml);
	}
	
	/* Create Pointers */
	cl_double *u, *v, *y, *phi,*L;
	
	costCalc *cost;
	cost = (costCalc *)malloc(S*sizeof(costCalc));
	
	dataset *data;
	data = (dataset *)malloc(S*sizeof(dataset));
	
	/* Assign memory to pointers based on scalars */
	// cost.u = (double *)malloc(sizeof(double)*(ml*ml+ml)/2);
	// cost.v = (double *)malloc(sizeof(double)*(ml*ml+ml)/2);
	// cost.y = (double *)malloc(sizeof(double)*(n-m+1));
	// cost.phi = (double *)malloc(sizeof(double)*((n-m+1)*(ml*ml+3*ml)/2));
	//phi = (double *)malloc(sizeof(double)*300*300);
	//L = (double *)malloc(sizeof(double)*300*300);
	
	/* Check malloc */
	if(v == NULL || u == NULL || y == NULL || phi == NULL)
	{
		mexPrintf("Variables [u, v, y, phi] were NOT allocated memory\n");
	}
	else if (mxIsStruct(prhs[4]))
	{
		/* Extract Vectors and copy data from MATLAB mex to C variables*/
        U = mxGetField(prhs[4], 0, "u_coords");
		memcpy(data->U, (double *)mxGetPr(U), sizeof(double)*(ml*ml+ml)/2);
		
		V = mxGetField(prhs[4], 0, "v_coords");
		memcpy(data->V, (double *)mxGetPr(V), sizeof(double)*(ml*ml+ml)/2);
		
		Y = mxGetField(prhs[4], 0, "y");
		memcpy(data->Y, (double *)mxGetPr(Y), sizeof(double)*(n-m+1));
		
		PHI = mxGetField(prhs[4], 0, "phi");
		memcpy(data->phi, (double *)mxGetPr(PHI), sizeof(double)*((n-m+1)*(ml*ml+3*ml)/2));
		
		//PHI = mxGetField(prhs[4], 0, "phi");
		//memcpy(phi, (double *)mxGetPr(PHI), sizeof(double)*300*300);
    }
		
	/* Find size of bytes required */
	size_t S_bytes = sizeof(double)*S;
	size_t d_bytes = sizeof(double)*d;
	
	/* Allocate additional host memory */
	double *h_gb = (double *)malloc(d_bytes);
	double h_gbcost;
	double *pdiff = (double *)malloc(S_bytes);
	double gdiff;
	
	/* Dynamically allocate memory for swarm */
	particle *swarm;
	swarm = (particle *)malloc(S*sizeof(particle));
	
	
	/////////////////////////////////FIND DEVICE device_id//////////////////////////////////////
	
	/* Define Device ID again from config()*/
	cl_platform_id cpPlatform;        	// OpenCL platform
	cl_device_id device_id;           	// device ID
	cl_int err;		
	size_t infoSize;
    cl_int MCU;
	/* Platform Setup */
	err = clGetPlatformIDs(1, &cpPlatform, NULL); 
	if(err != CL_SUCCESS ) {
		mexPrintf("Couldn't identify a platform"); getErrorString(err);
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
	
	/////////////////////////////////Compilation of Kernel//////////////////////////////////////
	
	cl_context context;               	// context
	cl_command_queue queue;          	// command queue
	cl_program program1, program2;    	// program
	cl_kernel kernel;					// kernel
	
	size_t localSize, globalSize;
	globalSize = S;	// Total Number of work items
	localSize = (size_t)ceil((double)globalSize/32.0);
	
	mexPrintf("Global Size = %d\n", globalSize);
	mexPrintf("Local Size = %d\n", localSize);
	
	/*Create a context */
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	
	/* Get source file...PSO.cl */
	FILE *source_handle,*binary_handle;
	char *source_buffer[NUM_FILES], *source_log, *binary_buffer, *binary_log;
	size_t source_size[NUM_FILES], log_size, binary_size;
	const char* source_files[]=SOURCE_FILES;
	/* Read each program file and place content into buffer */
	for (int i = 0; i < NUM_FILES; i++) {
		source_handle = fopen(source_files[i], "r");
		if(source_handle == NULL) {
			mexPrintf("Couldn't find the source file");
			return;
		}
		fseek(source_handle, 0, SEEK_END);
		source_size[i] = ftell(source_handle);
		rewind(source_handle);
		source_buffer[i] = (char*)malloc(source_size[i]+1);
		source_buffer[i][source_size[i]] = '\0';
		fread(source_buffer[i], sizeof(char), source_size[i], source_handle);
		fclose(source_handle);
	}
	
	/* SETUP KERNEL */
	/* Create a command Queue */
	queue = clCreateCommandQueue(context, device_id, 0, &err);
	if(err != CL_SUCCESS ) {
		mexPrintf("Couldn't create a command queue\n"); getErrorString(err);
		return;
	}
	
	/* Create program from source buffer */
	program1 = clCreateProgramWithSource(context, NUM_FILES, (const char**)&source_buffer, source_size, &err);
	if(err != CL_SUCCESS ) {
		mexPrintf("Couldn't create the program from source"); getErrorString(err);
		return;
	}
	
	for (int i = 0; i < NUM_FILES; i++) {
		free(source_buffer[i]);
	}
	
	/* Build program executable */
	/* Device_list is null: automatically selects the current device chosen */
	err=clBuildProgram(program1, 0, NULL, NULL, NULL, NULL);
	if(err != CL_SUCCESS ) {
		/* Find size of log and print to std output */
		mexPrintf("Build from source error\n"); getErrorString(err);
		clGetProgramBuildInfo(program1, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		source_log = (char*) malloc(log_size + 1);
		source_log[log_size] = '\0';
		clGetProgramBuildInfo(program1, device_id, CL_PROGRAM_BUILD_LOG, log_size + 1, source_log, NULL);
		source_log[log_size] = '\0';
		mexPrintf("%s\n", source_log);
		//free(source_log);
		return;
	} 
	
	/* Create Binary File */
	FILE *binFile;
	char *binary;
	clGetProgramInfo(program1, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binary_size, NULL);
	binary = (char*)malloc(binary_size);
	clGetProgramInfo(program1, CL_PROGRAM_BINARIES, binary_size, &binary, NULL);
	binFile = fopen(BINARY_FILE, "w");
	fwrite(binary, binary_size, 1, binFile);
	fclose(binFile);
	free(binary);
	//read the cache
	binary_handle = fopen(BINARY_FILE, "r");
	if(binary_handle == NULL) {
		mexPrintf("Couldn't find the binary file");
		return;
	}
	
	fseek(binary_handle, 0, SEEK_END);
	binary_size = ftell(binary_handle);
	rewind(binary_handle);
	binary_buffer = (char*)malloc(binary_size+1);
	binary_buffer[binary_size] = '\0';
	fread(binary_buffer, sizeof(char), binary_size, binary_handle);
	fclose(binary_handle);
	program2 = clCreateProgramWithBinary(context, 1, &device_id, &binary_size,(const unsigned char **)&binary_buffer, &binary_log, &err);
	free(binary_buffer);
	err=clBuildProgram(program2, 1, &device_id, NULL, NULL, NULL);
	if(err != CL_SUCCESS ) {
		mexPrintf("Build from cache error\n");
		return;
	}
	
	/////////////////////////////////Kernel Setup//////////////////////////////////////
	/* Create kernel */
	kernel = clCreateKernel(program2, "PPSO", &err);
	if(err != CL_SUCCESS ){mexPrintf("Error in clCreateKernel\n"); getErrorString(err); return;}
	
	cl_uint numOfKernels;
	err = clCreateKernelsInProgram(program2, 0, NULL, &numOfKernels);
	if (err != CL_SUCCESS) {
		 mexPrintf("Unable to retrieve kernel count from program\n");
	}else{ mexPrintf("# of Kernels: %d", (int)numOfKernels); }
	 
	/* Allocate buffers to kernel */
	/* Declare particle input/output buffers */
	cl_mem d_old;  
	cl_mem d_new;
	cl_mem d_gb;
	/* Declare boundaries */
	cl_mem d_ub;
	cl_mem d_lb;
	/* Declare random numbers */
	cl_mem d_r1;
	cl_mem d_r2;
	/* Declare cost*/
	cl_mem d_cost;
	/* Declare data bufer */
	cl_mem d_data;
	
	/* Create Cost buffer */
	d_cost = clCreateBuffer(context, CL_MEM_READ_WRITE , S*sizeof(costCalc), NULL, NULL);
	/* Create Cost buffer */
	d_data = clCreateBuffer(context, CL_MEM_READ_ONLY , sizeof(dataset), NULL, NULL);
	/* Create input particle buffer*/
	d_old = clCreateBuffer(context, CL_MEM_READ_ONLY , S*sizeof(particle), NULL, NULL);
	/* Create output particle buffer*/
	d_new = clCreateBuffer(context, CL_MEM_WRITE_ONLY , S*sizeof(particle), NULL, NULL);
	/* Create global best position */
	d_gb = clCreateBuffer(context, CL_MEM_READ_ONLY , d_bytes, NULL, NULL);
	/* Create boundary buffer*/
	d_ub = clCreateBuffer(context, CL_MEM_READ_ONLY , d_bytes, NULL, NULL);
	d_lb = clCreateBuffer(context, CL_MEM_READ_ONLY , d_bytes, NULL, NULL);
	/* Create random number buffer */
	d_r1 = clCreateBuffer(context, CL_MEM_READ_ONLY , S_bytes*d, NULL, NULL);
	d_r2 = clCreateBuffer(context, CL_MEM_READ_ONLY , S_bytes*d, NULL, NULL);
	
	/* Set Arguments for kernel */
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_cost);
	if(err != CL_SUCCESS ){mexPrintf("Failure arg 0\n"); getErrorString(err); return;}
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_data);
	if(err != CL_SUCCESS ){mexPrintf("Failure arg 0\n"); getErrorString(err); return;}
	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_old);
	if(err != CL_SUCCESS ){mexPrintf("Failure arg 1\n"); getErrorString(err); return;}
	err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_new);
	if(err != CL_SUCCESS ){mexPrintf("Failure arg 2\n"); getErrorString(err); return;} 
	err = clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_gb);
	if(err != CL_SUCCESS ){mexPrintf("Failure arg 3\n"); getErrorString(err); return;} 
	err = clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_ub);
	if(err != CL_SUCCESS ){mexPrintf("Failure arg 4\n"); getErrorString(err); return;}
	err = clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_lb);
	if(err != CL_SUCCESS ){mexPrintf("Failure arg 5\n"); getErrorString(err); return;}
	err = clSetKernelArg(kernel, 7, sizeof(cl_mem), &d_r1);
	if(err != CL_SUCCESS ){mexPrintf("Failure arg 6\n"); getErrorString(err); return;}
	err = clSetKernelArg(kernel, 8, sizeof(cl_mem), &d_r2);
	if(err != CL_SUCCESS ){mexPrintf("Failure arg 7\n"); getErrorString(err); return;}
	err = clSetKernelArg(kernel, 10, sizeof(int), &S);
	if(err != CL_SUCCESS ){mexPrintf("Failure arg 9\n"); getErrorString(err); return;}
	err = clSetKernelArg(kernel, 11, sizeof(int), &d);
	if(err != CL_SUCCESS ){mexPrintf("Failure arg 10\n"); getErrorString(err); return;}
	
	///////////////////////////////////////////////////////////////////
	/* Malloc random variables - both float and double */
	//float *R = (float *)malloc(sizeof(float)*(S*d));
	float *R =  (float *)malloc(sizeof(float)*(2));
	double *r1 = (double *)malloc(sizeof(double)*(S*d));
	double *r2 = (double *)malloc(sizeof(double)*(S*d));
	
	/* Open random generated file */
	fileID = fopen("rand.txt", "r");
	
	/* Initialise position & pbest & velocity */
	for(i = 0; i<S ;i++)
	{
		for(j = 0; j<d ;j++)
		{
			/* Scan random variables from file ... fscanf() uses FLOATS */
			fscanf(fileID, "%f %f", &R[0], &R[1]);
			/* Convert values to double percision - 16 -> 32 bit */
			r1[i+(j*S)] = (double)R[0];
			/* Convert values to double percision - 16 -> 32 bit */
			r2[i+(j*S)] = (double)R[1];
			
			/* Set jth position randomly (lower limit < position <upper limit) */
			swarm[i].position[j] = l_limits[j]+ r1[i+(j*S)]*(u_limits[j]-l_limits[j]);
			swarm[i].pbest[j] = swarm[i].position[j];
			
			/* Set particels wth random velocities */
			swarm[i].velocity[j] = fabs(u_limits[j]-l_limits[j])/10.0*r1[i+(j*S)];
			swarm[i].velocity[j] = (swarm[i].velocity[j] - fabs(u_limits[j]-l_limits[j])/10.0*r2[i+(j*S)]);
			
		}
	}
	
	swarm[0].position[0] = 1;
	swarm[0].position[1] = 0.7;
	swarm[0].position[2] = 1;
	swarm[0].position[3] = 0.7;
	swarm[0].position[4] = 0.7;
	swarm[0].position[5] = 10;
	
	/////////////////////////////////PSO & KERNEL EXECUTION////////////////////////////////////
	
	/* Loop counter */
	double k1 = 0;
	/* Inertia constant */
	double w = 0.7;
	/* Social and Cognitive Acceleration factors */
	double C = 1.2;
	
	/* Flag variables */
	int flag1 = 0; //Raised when position exceeds boundaries
	int flag2 = 1; //Raised when veolicty exceeds boundaries
	int id = 0;
	
	int ml_2 = (ml*ml+ml)/2;
	double temp;
	
	int n_size = n-m+1;
	
	/* FORWARD SUB AND BACK SUB BUFFERS */
	double *x_1;
	double *x_2;
	x_1 = (double *)malloc(sizeof(double)*(n-m+1));
	x_2 = (double *)malloc(sizeof(double)*(n-m+1));
	
	mexPrintf("Samples: %d, Memory Length 1 order: %d, Memory Length 2 order: %d  \n", n, ml, ml_2);
	/////////////////////PSO////////////////////
	err = clEnqueueWriteBuffer(queue, d_data, CL_TRUE, 0, sizeof(dataset), data, 0, NULL, NULL);
	if(err != CL_SUCCESS ){mexPrintf("Writing Failure 0.1\n"); getErrorString(err);}
	err = clEnqueueWriteBuffer(queue, d_ub, CL_TRUE, 0, d_bytes, u_limits, 0, NULL, NULL);
	if(err != CL_SUCCESS ){mexPrintf("Writing Failure 0.2\n"); getErrorString(err);}
	err = clEnqueueWriteBuffer(queue, d_lb, CL_TRUE, 0, d_bytes, l_limits, 0, NULL, NULL);
	if(err != CL_SUCCESS ){mexPrintf("Writing Failure 0.3\n"); getErrorString(err); }
		
	while(flag2)
	{
		/* Write structure into input in device memory */
		err = clEnqueueWriteBuffer(queue, d_r1, CL_TRUE, 0, S_bytes*d, r1, 0, NULL, NULL);
		if(err != CL_SUCCESS ){mexPrintf("Writing Failure 1\n"); getErrorString(err); break;}
		err = clEnqueueWriteBuffer(queue, d_r2, CL_TRUE, 0, S_bytes*d, r2, 0, NULL, NULL);
		if(err != CL_SUCCESS ){mexPrintf("Writing Failure 2\n"); getErrorString(err); break;}
		err = clEnqueueWriteBuffer(queue, d_gb, CL_TRUE, 0, d_bytes, h_gb, 0, NULL, NULL);
		if(err != CL_SUCCESS ){mexPrintf("Writing Failure 3\n"); getErrorString(err); break;}
		err = clEnqueueWriteBuffer(queue, d_cost, CL_TRUE, 0, S*sizeof(costCalc), cost, 0, NULL, NULL);
		if(err != CL_SUCCESS ){mexPrintf("Writing Failure 4:\n");getErrorString(err);  break;}
		err = clEnqueueWriteBuffer(queue, d_old, CL_TRUE, 0, S*sizeof(particle), swarm, 0, NULL, NULL);
		if(err != CL_SUCCESS ){mexPrintf("Writing Failure 5\n"); getErrorString(err); break;}
		err = clSetKernelArg(kernel, 9, sizeof(double), &k1);
		if(err != CL_SUCCESS ){mexPrintf("Writing Failure 6\n");getErrorString(err);  break;}
		
		/* Execute Kernel */
		err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize,0, NULL, NULL);
		if(err != CL_SUCCESS ){mexPrintf("Execute Failure\n");getErrorString(err); break;}
		/* Wait for queue before reading results back in */
		clFinish(queue);
		
		/* Read Results */
		err = clEnqueueReadBuffer(queue, d_new, CL_TRUE, 0, S*sizeof(particle), swarm, 0, NULL, NULL );
		if(err != CL_SUCCESS ){mexPrintf("Read Failure\n"); getErrorString(err); break;}
		
		/* Evaluate particle best */
		/* Initilialise global best position & cost as 1st particle in 1st iteration */
		if(k1 == 0)
		{
			h_gbcost = swarm[0].pcost;
			for (j = 0; j < d; j++)
			{
				h_gb[j] = swarm[0].position[j];
			}
		}
		
		dimensions[0] =0;
		dimensions[1] =0;
		dimensions[2] =0;
		dimensions[3] =0;
		dimensions[4] =0;
		dimensions[5] =0;
		
		/* Evaluate global best: h_gb and h_gbcost*/
        for(i = 0;i<S;i++)
		{
			/* Find best global position in swarm */
			if(swarm[i].pcost < h_gbcost) 
			{
					for (j = 0; j < d;j++) 
					{
						h_gb[j] = swarm[i].pbest[j];
					}
					
					h_gbcost = swarm[i].pcost;
					
            }
			
			/* Find difference between best known particle costs */
			for(j = 0; j<d ; j++){
				if(j == 0)
				{
					lock = 0;
					dimensions[j] = (swarm[i].position[j]- h_gb[j])*(swarm[i].position[j]- h_gb[j])/(double)S;
				}
				else
				{
					dimensions[j] = dimensions[j] + (swarm[i].position[j]- h_gb[j])*(swarm[i].position[j]- h_gb[j])/(double)S;
				}
			}
        }
		
		k1 = k1+1;
		/* Termination Criterion */
		/* Maximum Iterations */
		
		
		/* Lack of change (cost) */
		for(j = 0; j<d ; j++)
		{
			if(sqrt(dimensions[j]) < maxCost)
			{
				lock = lock + 1;
				mexPrintf("Dimensions %d: %f\n", j, sqrt(dimensions[j]));
			}
		}
		
		if(k1 == maxIter)
		{
			mexPrintf("Reached maximum iterations\n");
			//If maximum iterations are exceeded..i should have a percentage confidence...
			//TODO
			flag2 = 0;
			
		}
		
		if(lock == d)
		{
			mexPrintf("Particles are pretty close\n");
			//If maximum iterations are exceeded..i should have a percentage confidence...
			//TODO
			flag2 = 0;
		}
		
		/* Linearly Decrease inerrtia constant - w */
		if(flag2)
		{
			/* Continuously Scan rand vars from file --> easiest way to acheive random numbers upon every iteration in C */
			for(i = 0; i<S ;i++)
			{
				for(j = 0; j<d ;j++)
				{
					/* Scan random variables from file ... fscanf() uses FLOATS */
					fscanf(fileID, "%f %f", &R[0], &R[1]);
					/* Convert values to double percision - 16 -> 32 bit */
					r1[i+(j*S)] = (double)R[0];
					/* Convert values to double percision - 16 -> 32 bit */
					r2[i+(j*S)] = (double)R[1];
				}
			}
		}
	}
	
	/* Return */
	plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(1, (mwSize)d, mxREAL);
    memcpy(mxGetPr(plhs[0]), &h_gbcost, sizeof(double));
	memcpy(mxGetPr(plhs[1]), h_gb, d_bytes);
	
	/* Free the host from its gloriuos evaluation */
	free(h_gb);
	// free(phi);
	// free(y);
	// free(u);
	// free(v);
	free(r1);
	free(r2);
	free(R);
	
	/* Delete device info for space */
	clReleaseMemObject(d_old);  
	clReleaseMemObject(d_new);
	clReleaseMemObject(d_gb);
	/* Declare boundaries */
	clReleaseMemObject(d_ub);
	clReleaseMemObject(d_lb);
	/* Declare random numbers */
	clReleaseMemObject(d_r1);
	clReleaseMemObject(d_r2);
	/* Realese CostF */
	clReleaseMemObject(d_cost);  
	
	/* Release the kernel and program from its deeds */
	clReleaseProgram(program1);
	clReleaseProgram(program2);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
	
	/* Close files */
	fclose(fileID);
	mexPrintf("MEX Complete. Good Job\n");
}

/* Decomp tests in mex-c */
int CHOL(double A[], double L[], int N){
	
	/* Iterative variables */
	int i, j, k;
	
	/* Set flag 1/0: checks positive definiteness */
	int FLAG = 0;
	
	/* Temporary Summing variable */
	double sumEntries = 0;
	
	/* Create L = zeroes(N) */
	for (i = 0; i < N ; i++)
	{
		for (j = 0; j < N; j++)
		{
			L[i*N+j] = 0;
		}
	}
	
	/* Find L using chol alogrithm*/
	for (j = 0; j < N ; j++)
	{
		for (i = j; i < N ; i++)
		{
			/* Reset SumEntries */
			sumEntries = 0;
			
			/* Check for diagnonal based on chol alg */
			if(i == j)
			{
				//mexPrintf("i: %d j: %d\t", i, j);
				/* Find diagonal component */
				for(k = 0; k < j; k++)
				{
					//mexPrintf("%d\t%d \t", k,j);
					sumEntries = sumEntries + pow(L[j*N+k],2);
				}
				/* Check Flag */
				if( A[i*N+j] < sumEntries)
				{
					/* Raise flag and return error*/
					//mexPrintf("You ffed up : %f\t %f \n", A[i*N+j], sumEntries);
					FLAG = 1;
					return 1;
				}
				
				/* Must not be negative */
				L[i*N+j] = sqrt(A[i*N+j]-sumEntries);
				
			}
			else /* Non Diagonal terms */
			{
				for(k = 0;k<j;k++)
				{
					sumEntries = sumEntries + L[j*N+k]*L[i*N+k];
				}
				
				L[i*N+j] = 1/L[j*N+j]*(A[i*N+j]-sumEntries);
			}
		}
	}
	
	/* Return 0 if complete */
	return 0;
	
}

void SquareMatrixMul(double A[], double B[], double C[], int N){
	
	int i, j, k;
	double temp = 0;
	
	for(i = 0; i < N; i++)
	{
		for(j = 0; j < N; j++)
		{
			temp = 0;
			
			for(k = 0; k < N; k++)
			{
				temp = temp + A[i + k*N]*B[j + k*N];
			}
			
			C[i*N + j] = temp;
		}
	}
}

void getErrorString(cl_int error)
{
switch(error){
    // run-time and JIT compiler errors
    case 0:  mexPrintf("CL_SUCCESS \n"); return;
    case -1:  mexPrintf("CL_DEVICE_NOT_FOUND\n"); return;
    case -2:  mexPrintf("CL_DEVICE_NOT_AVAILABLE\n"); return;
    case -3:  mexPrintf("CL_COMPILER_NOT_AVAILABLE\n");return;
    case -4:  mexPrintf("CL_MEM_OBJECT_ALLOCATION_FAILURE\n");return;
    case -5:  mexPrintf("CL_OUT_OF_RESOURCES\n");return;
    case -6:  mexPrintf("CL_OUT_OF_HOST_MEMORY\n");return;
    case -7:  mexPrintf("CL_PROFILING_INFO_NOT_AVAILABLE\n");return;
    case -8:  mexPrintf("CL_MEM_COPY_OVERLAP\n");return;
    case -9:  mexPrintf("CL_IMAGE_FORMAT_MISMATCH\n");return;
    case -10:  mexPrintf("CL_IMAGE_FORMAT_NOT_SUPPORTED\n");return;
    case -11:  mexPrintf("CL_BUILD_PROGRAM_FAILURE\n");return;
    case -12:  mexPrintf("CL_MAP_FAILURE\n");return;
    case -13:  mexPrintf("CL_MISALIGNED_SUB_BUFFER_OFFSET\n");return;
    case -14:  mexPrintf("CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST\n");return;
    case -15:  mexPrintf("CL_COMPILE_PROGRAM_FAILURE\n");return;
    case -16:  mexPrintf("CL_LINKER_NOT_AVAILABLE\n");return;
    case -17:  mexPrintf("CL_LINK_PROGRAM_FAILURE\n");return;
    case -18:  mexPrintf("CL_DEVICE_PARTITION_FAILED\n");return;
    case -19:  mexPrintf("CL_KERNEL_ARG_INFO_NOT_AVAILABLE\n");return;

    // compile-time errors
    case -30:  mexPrintf("CL_INVALID_VALUE\n");return;
    case -31:  mexPrintf("CL_INVALID_DEVICE_TYPE\n");return;
    case -32:  mexPrintf("CL_INVALID_PLATFORM\n");return;
    case -33:  mexPrintf("CL_INVALID_DEVICE\n");return;
    case -34:  mexPrintf("CL_INVALID_CONTEXT\n");return;
    case -35:  mexPrintf("CL_INVALID_QUEUE_PROPERTIES\n");return;
    case -36:  mexPrintf("CL_INVALID_COMMAND_QUEUE\n");return;
    case -37:  mexPrintf("CL_INVALID_HOST_PTR\n");return;
    case -38:  mexPrintf("CL_INVALID_MEM_OBJECT\n");return;
    case -39:  mexPrintf("CL_INVALID_IMAGE_FORMAT_DESCRIPTOR\n");return;
    case -40:  mexPrintf("CL_INVALID_IMAGE_SIZE\n");return;
    case -41:  mexPrintf("CL_INVALID_SAMPLER\n");return;
    case -42:  mexPrintf("CL_INVALID_BINARY\n");return;
    case -43:  mexPrintf("CL_INVALID_BUILD_OPTIONS\n");return;
    case -44:  mexPrintf("CL_INVALID_PROGRAM\n");return;
    case -45:  mexPrintf("CL_INVALID_PROGRAM_EXECUTABLE\n");return;
    case -46:  mexPrintf("CL_INVALID_KERNEL_NAME\n");return;
    case -47:  mexPrintf("CL_INVALID_KERNEL_DEFINITION\n");return;
    case -48:  mexPrintf("CL_INVALID_KERNEL\n");return;
    case -49:  mexPrintf("CL_INVALID_ARG_INDEX\n");return;
    case -50:  mexPrintf("CL_INVALID_ARG_VALUE\n");return;
    case -51:  mexPrintf("CL_INVALID_ARG_SIZE\n");return;
    case -52:  mexPrintf("CL_INVALID_KERNEL_ARGS\n");return;
    case -53:  mexPrintf("CL_INVALID_WORK_DIMENSION\n");return;
    case -54:  mexPrintf("CL_INVALID_WORK_GROUP_SIZE\n");return;
    case -55:  mexPrintf("CL_INVALID_WORK_ITEM_SIZE\n");return;
    case -56:  mexPrintf("CL_INVALID_GLOBAL_OFFSET\n");return;
    case -57:  mexPrintf("CL_INVALID_EVENT_WAIT_LIST\n");return;
    case -58:  mexPrintf("CL_INVALID_EVENT\n");return;
    case -59:  mexPrintf("CL_INVALID_OPERATION\n");return;
    case -60:  mexPrintf("CL_INVALID_GL_OBJECT\n");return;
    case -61:  mexPrintf("CL_INVALID_BUFFER_SIZE\n");return;
    case -62:  mexPrintf("CL_INVALID_MIP_LEVEL\n");return;
    case -63:  mexPrintf("CL_INVALID_GLOBAL_WORK_SIZE\n");return;
    case -64:  mexPrintf("CL_INVALID_PROPERTY\n");return;
    case -65:  mexPrintf("CL_INVALID_IMAGE_DESCRIPTOR\n");return;
    case -66:  mexPrintf("CL_INVALID_COMPILER_OPTIONS\n");return;
    case -67:  mexPrintf("CL_INVALID_LINKER_OPTIONS\n");return;
    case -68:  mexPrintf("CL_INVALID_DEVICE_PARTITION_COUNT\n");return;

    // extension errors
    case -1000:  mexPrintf("CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR\n");return;
    case -1001:  mexPrintf("CL_PLATFORM_NOT_FOUND_KHR\n");return;
    case -1002:  mexPrintf("CL_INVALID_D3D10_DEVICE_KHR\n");return;
    case -1003:  mexPrintf("CL_INVALID_D3D10_RESOURCE_KHR\n");return;
    case -1004:  mexPrintf("CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR\n");return;
    case -1005:  mexPrintf("CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR\n");return;
    default:  mexPrintf("Unknown OpenCL error\n");return;
    }
}