// OpenCL kernel. Each work item takes care of one element of c
//Useful in built functions...
//	-> fabs(); floating point absolute value
//  -> sqrt(); square root of value
//Functions
//	-> TCfunc_new(); calculates function


//Enable inbuilt library
#if __OPENCL_VERSION__ <= CL_VERSION_1_1
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

/* Define Particle Properties - 2nd order only*/
typedef struct
{
	/* PSO */
	double position[6];
	double pcost;
	double velocity[6];
	double pbest[6];
	double pbcost;
	
}particle;

typedef struct
{
	
	/* Cost - ml = 12, N = 400, n = 70 */
	double prior1[12*12];
	double prior2[78*78];
	double sig[731*731];
	double Q[731*731];
	
}costCalc;

typedef struct
{
	double phi[731*90];
	double U[78];
	double V[78];
	double Y[731];
}dataset;

/* Cholesky Factorization Function */
int CHOL(__global double A[], __global double L[], unsigned int N){
	
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
					sumEntries = sumEntries + L[j*N+k]*L[j*N+k];
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

/* Calculates Cost */
double func(__global costCalc *cost, __global dataset *data, __global double hyper[], 
				unsigned int N, unsigned int n, unsigned int ml, unsigned int id) {
	
	int i, j, k;
	
	int ml_2, n_size;
	
	ml_2 = (ml+ml*ml)/2;
	
	n_size = N-n+1;
	
	/* Cholesky Flag */
	int flag1;
	flag1 = 0;
	
	double temp;
	
	/* Create Prior One */
	for(i = 0; i < ml; i++)
	{
		for(j = 0; j < ml; j++)
		{
			cost[id].prior1[i*ml+j] = hyper[0]*pow(hyper[1], max(i+1, j+1));
		}
		
		/* Add small tolernace to diagonals-  */
		if(i == j){
			cost[id].prior1[(i)*ml+(j)] = cost[id].prior1[(i)*ml+(j)]+0.000001;
		}
	}
		
	
	/* Create Prior Two */
	for(i = 0; i < ml_2; i++)
	{
		for(j = 0; j < ml_2; j++)
		{
			cost[id].prior2[(i)*ml_2+(j)] = pow(hyper[3], max(data->V[i], data->V[j]));

			cost[id].prior2[(i)*ml_2+(j)] = cost[id].prior2[(i)*ml_2+(j)]*pow(hyper[4], max(data->U[i], data->U[j]));
			
			cost[id].prior2[(i)*ml_2+(j)] = hyper[2]*cost[id].prior2[(i)*ml_2+(j)];
			
			cost[id].prior2[(i)+(j)*ml_2] = cost[id].prior2[(i)*ml_2+(j)];
			
			/* Add small tolernace to diagonals: Positive Definiteness  */
			
			if(i == j)
			{
				cost[id].prior2[(i)*ml_2+(j)] = cost[id].prior2[(i)*ml_2+(j)]+0.000001;
			}
		}
	}
	
	
	for(i = 0; i < ml+ml_2; i++)
	{
		for(j = 0; j < n_size; j++)
		{
			/* Reset temporary variable */
			temp = 0;
			
			/* Split Priors: Note that now for loops are minimized from ml+ml_2 to ml and ml_2  */
			
			if(i<ml)
			{
				
				for(k = 0; k < ml; k++)
				{
					temp = temp + cost[id].prior1[i*ml + k]*data->phi[j + k*n_size];
				}
				
			}else{
				
				for(k = 0; k < ml_2; k++)
				{
					temp = temp + cost[id].prior2[(i-ml)*ml_2 + k]*data->phi[j + (k+ml)*n_size];
				}
				
			}
			
			cost[id].Q[i*n_size + j] = temp;
		}
	}
		
	/* SIG = PHI'*P*PHI + noise_std^2*eye(length(SIG))- Evaluate covariance matrix*/
	for(i = 0; i < n_size; i++)
	{
		for(j = 0; j < n_size; j++)
		{
			temp = 0;
			
			for(k = 0; k < ml+ml_2; k++)
			{
				temp = temp + data->phi[i + k*n_size]*cost[id].Q[j + k*n_size];
			}
			
			cost[id].sig[i*n_size + j] = temp;
			if(i == j)
			{
				cost[id].sig[i*n_size + j] = cost[id].sig[i*n_size + j] + hyper[5]*hyper[5];
			}
		}
	
	}
			
	flag1 = CHOL(cost[id].sig, cost[id].Q, n_size);
	
	if(!flag1)
	{
		/*Reuse Sig*/
		double det;
		det = 0;
		for (j = 0; j < n_size; j++) 
		{
			/* x1 */
			cost[id].sig[j] = 0;
			/* x2 */
			cost[id].sig[n_size+j] = 0;
		}
		
		//Lx_1 = Y --> back substitution (lower triangle)
		//Solve x_1 = 1/L * Y
		
		for (j = 0; j < n_size; j++) 
		{
			temp = 0;

			for (i = 0; i < j; i++)
			{
				//temp = cost[id].Q[j*n_size + i] * x_1[i] + temp;
				temp = cost[id].Q[j*n_size + i] * cost[id].sig[j] + temp;
			}

			//x_1[j] = (cost[id].y[j] - temp) / (cost[id].Q[j * n_size + j]);
			cost[id].sig[j] = (data->Y[j] - temp) / (cost[id].Q[j * n_size + j]);
		}
		
		//Ux_2 = x_1 --> back substitution (upper triangle)
		//Solve x_2 = 1/U * x_1
		
		for (j = n_size - 1; j >= 0; j--) 
		{
			temp = 0;

			for (i = n_size - 1; i > j; i--)
			{
				//temp = cost[id].Q[i*n_size + j] * x_2[i] + temp;
				temp = cost[id].Q[i*n_size + j] * cost[id].sig[n_size+j] + temp;
			}
			//x_2[j] = (x_1[j] - temp) / (cost[id].Q[i * n_size + i]);
			cost[id].sig[n_size+j] = (cost[id].sig[j] - temp) / (cost[id].Q[i * n_size + i]);
		}
		
		/*
		det = log(det(LU)) == log(det(L)*det(U)) --> det(triangle) = product of diagonals
		det(U) == 1, det(L) ==> log(det(L)*det(U)) == prod(eig(L))
		logdet(prod(eig(L))) == sum(log(diag(L)))
		if U = L', then det(LU) = 2*det(L) 
		*/

		//J = Y'x_2 + det
		temp = 0;
		det = 0;
		for (j = 0; j < n_size; j++) 
		{
			//temp = temp + cost[id].Y[j]*x_2[j];
			temp = temp + data->Y[j]*cost[id].sig[n_size+j];
			det = det + log((cost[id].Q[j*n_size+j]));
		}
		
		return temp + 2*det;
	}
	else
	{
		//Illegal cost: set high or set negative, better option would be to return a NAN or INF
		return 200000.00;
	}
	
}

__kernel void PPSO(__global costCalc *costF,
					__global dataset *data,
					__global particle *past,
                    __global particle *future,
                    __global double *gb,
                    __global double *ub,
                    __global double *lb,
                    __global double *r1,
					__global double *r2,
					double k1,
                    unsigned int S,
                    unsigned int d){
	
    //Extracts Particle ID
    int id = get_global_id(0);
    //Particle dimension
    int j = 0;

    //Particle Adjustments
    double w, c;
    w = 1;
    c = 1.2;

	//Velocity, position and cost holder
	double v = 0;
	double p = 0;
	double cost = 0;
	
    //Uses kernel configuration to loop through swarm size
    if(id<S){
	
		// Linearly Decrease w or you can keep it constant
		if(w>0.7){
			//Comment one line at a time and see how much it chnages the results- looking at a factor of 10!!
			w = 1-k1/100;
		}
	
		// Initialisation (first iteration)
		if(k1 == 0){
			// Initialise cost and return old info
			for(j=0;j<d;j++)
			{
				future[id].position[j] = past[id].position[j];
				future[id].velocity[j] = past[id].velocity[j];
				future[id].pbest[j] = past[id].pbest[j];
			}
			
			
			// double func(__global double Y[], __global double phi[], 
			// __global double u[], __global double v[],
			// __global double prior1[], __global double prior2[],
			// __global double hyper[], unsigned int N, unsigned int n, unsigned int ml) {
				
			future[id].pcost = func(costF, data, past[id].position, 800, 70, 12, id);
		}
		else
		{
	
			for(j=0;j<d;j++){
			
				//Update velocity 
				v = w*past[id].velocity[j]+c*r1[id]*(past[id].pbest[j]-past[id].position[j])+c*r2[id+S]*(gb[j] - past[id].position[j]);
			
				//Check Velocity Boundaries- redirect if too large
				if(fabs(v) > fabs(ub[j]-lb[j])/5){
				
					if(v > 0)
					{
						v = (ub[j]-lb[j])/5;
					}
					else
					{
						v = -(ub[j]-lb[j])/5;
					}
					
				}
				
				future[id].velocity[j] = v;
				
				//Update Position
				if(p + v >= ub[j])
				{
					p = ub[j] - 0.000001;
					future[id].velocity[j] = -v;
				}
				else if(p + v <= lb[j])
				{
					p = lb[j] + 0.000001;
					future[id].velocity[j] = -v;
				}
				else
				{
					p = past[id].position[j] + v;
				}
				
				future[id].position[j] = p;
				
				// Update future pbest as future position
				future[id].pbest[j] = p;
			}
			
			// Update future particle best known cost as current position cost
			cost = func(costF, data, past[id].position, 800, 70, 12, id);
			
			/* Update Particles cost if better */
			if(cost < past[id].pbcost || k1 == 0)
			{
				future[id].pbcost = cost;
				
				for (j = 0; j < d;j++) 
				{
					future[id].pbest[j] = past[id].position[j];
				}
			}
			
			// Test whether current particle position is actually better than previous position
			// If it is not, revert back to old position
			if (cost > past[id].pbcost)
			{
				for(j=0;j<d;j++)
				{
					//Dupe particle best cost
					future[id].pbest[j] = past[id].pbest[j];
				}
				
				//Revert back to old cost
				future[id].pbcost = past[id].pbcost;
			}
			else
			{
				// Finalise futre cost if it is better
				future[id].pcost = cost;
			}
		}
    }
	
	return;
} 
