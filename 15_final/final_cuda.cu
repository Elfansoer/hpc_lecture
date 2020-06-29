/*
Fathul Asrar Alfansuri
19M38053

2D Navier-Stokes code written in C++, parallelized using CUDA.
The result of this code is 2 files, 'u.txt' and 'v.txt', a comma-separated data file.
NOTE: The code wasn't tested because of unsupported environment (cudaLaunchCooperativeKernel returns error code 71: not supported)
*/

#include <stdio.h>
#include <cstdio>
#include <cstdlib>
#include <cooperative_groups.h>
using namespace cooperative_groups;

// consts
int nt = 500;
int nit = 50;
const int nx = 41;
const int ny = 41;
__constant__ double dx = 2.0 / (nx - 1);
__constant__ double dy = 2.0 / (ny - 1);
__constant__ double rho = 1;
__constant__ double nu = 0.1;
__constant__ double dt = 0.001;

// maps

// copy buffers

// Build up equations
__device__ double build_up_b( int i, int j, double *u, double *v ) {
	double term1 = ( u[j*nx+(i+1)]-u[j*nx+(i-1)] )/( 2*dx );
	double term2 = ( v[(j+1)*nx+i]-v[(j-1)*nx+i] )/( 2*dy );

	double temp1 = (term1+term2)/dt;
	double temp2 = term1*term1;
	double temp4 = term2*term2;

	double temp31 = ( u[(j+1)*nx+i]-u[(j-1)*nx+i] )/( 2*dy );
	double temp32 = ( v[j*nx+(i+1)]-v[j*nx+(i-1)] )/( 2*dx );
	double temp3 = 2*temp31*temp32;

	double temp5 = rho * (temp1 - temp2 - temp3 - temp4);

	return temp5;
}

// Pressure poisson
__device__ void pressure_poisson( int i, int j, double b, double *p, double *pnext ) {
	// don't do anything on edges
	if (i==0 || j==0 || i==nx-1 || j==ny-1) {
		pnext[j*nx+i] = 0;
		return;
	}

	double temp11 = ( p[j*nx+(i+1)]+p[j*nx+(i-1)] ) * dy*dy;
	double temp12 = ( p[(j+1)*nx+i]+p[(j-1)*nx+i] ) * dx*dx;
	double temp1 = temp11+temp12;

	double temp2 = 2 * ( dx*dx + dy*dy );

	double temp31 = dx*dx * dy*dy;
	double temp32 = 2 * ( dx*dx + dy*dy );
	double temp3 = temp31/temp32 * b;

	double temp4 = temp1/temp2 - temp3;

	pnext[j*nx+i] = temp4;
}
__device__ void pressure_poisson_2( int i, int j, double *pnext ) {
	// calculate edges after synchronized
	if (i==0) {
		pnext[j*nx+i] = pnext[j*nx+(i+1)];
	}
	if (j==0) {
		pnext[j*nx+i] = pnext[(j+1)*nx+i];
	}
	if (i==nx-1) {
		pnext[j*nx+i] = pnext[j*nx+(i-1)];
	}
	if (j==ny-1) {
		pnext[j*nx+i] = 0;
	}
}

// Cavity flow
__device__ void cavity_flow_pre( int i, int j, double *unext, double *vnext ) {
	unext[j*nx+i] = (j==ny-1)*1;
	vnext[j*nx+i] = 0;
}
__device__ void cavity_flow( int i, int j, double *u, double *v, double *p, double *unext, double *vnext ) {
	// u momentum
	double temp1 = u[j*nx+i];
	double temp2 = u[j*nx+i] * dt/dx * ( u[j*nx+i]-u[j*nx+(i-1)] );
	double temp3 = v[j*nx+i] * dt/dy * ( u[j*nx+i]-u[(j-1)*nx+i] );
	double temp4 = dt/( 2*rho*dx ) * ( p[j*nx+(i+1)]-p[j*nx+(i-1)] );

	double temp51 = dt/(dx*dx) * ( u[j*nx+(i+1)] - 2*u[j*nx+i] + u[j*nx+(i-1)] );
	double temp52 = dt/(dy*dy) * ( u[(j+1)*nx+i] - 2*u[j*nx+i] + u[(j-1)*nx+i] );
	double temp5 = nu * (temp51+temp52);

	double temp6 = temp1 - temp2 - temp3 - temp4 + temp5;
	unext[j*nx+i] = temp6;

	// v momentum
	temp1 = v[j*nx+i];
	temp2 = u[j*nx+i] * dt/dx * ( v[j*nx+i]-v[j*nx+(i-1)] );
	temp3 = v[j*nx+i] * dt/dy * ( v[j*nx+i]-v[(j-1)*nx+i] );
	temp4 = dt/( 2*rho*dy ) * ( p[(j+1)*nx+i]-p[(j-1)*nx+i] );

	temp51 = dt/(dx*dx) * ( v[j*nx+(i+1)] - 2*v[j*nx+i] + v[j*nx+(i-1)] );
	temp52 = dt/(dy*dy) * ( v[(j+1)*nx+i] - 2*v[j*nx+i] + v[(j-1)*nx+i] );
	temp5 = nu * (temp51+temp52);

	temp6 = temp1 - temp2 - temp3 - temp4 + temp5;
	vnext[j*nx+i] = temp6;
}

// cuda kernel
__global__ void kernel( int nt, int nit, double *u, double *v, double *p, double *unext, double *vnext, double *pnext, double *temp ) {
	// init identifier
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	grid_group grid = this_grid();

	if (i>=nx) return;
	if (j>=ny) return;

	// init map
	unext[j*nx+i] = 0;
	vnext[j*nx+i] = 0;
	pnext[j*nx+i] = 0;
	temp[j*nx+i] = 100; // for debugging
	grid.sync();
	// __syncthreads();

	// main iteration
	for( int ctr1=0; ctr1<nt; ctr1++ ) {
		// create copy
		u[j*nx+i] = unext[j*nx+i];
		v[j*nx+i] = vnext[j*nx+i];
		grid.sync();
		// __syncthreads();

		// build up b
		double b = build_up_b( i, j, u, v );
		grid.sync();
		// __syncthreads();

		// sub-iteration
		for( int ctr2=0; ctr2<nit; ctr2++ ) {
			// create copy
			p[j*nx+i] = pnext[j*nx+i];
			grid.sync();
			// __syncthreads();

			// pressure poisson
			pressure_poisson( i, j, b, p, pnext );
			grid.sync();
			// __syncthreads();

			// resolve edges
			pressure_poisson_2( i, j, pnext );
			grid.sync();
			// __syncthreads();
		}

		// cavity flow init value, especially edges
		cavity_flow_pre( i, j, unext, vnext );
		grid.sync();
		// __syncthreads();

		// real cavity flow
		cavity_flow( i, j, u, v, pnext, unext, vnext );
		grid.sync();
		// __syncthreads();
	}
}

int main() {
	// inits
	nt = 300;
	nit = 50;
	int size = nx * ny * sizeof( double );
	int nb = 32;
	int ngx = 1 + nx/nb;
	int ngy = 1 + ny/nb;
	if (nx%nb==0) ngx--;
	if (ny%nb==0) ngy--;

	// host
	double *u, *v;
	u = new double[nx*ny];
	v = new double[nx*ny];

	// device
	double *ud, *vd, *pd;
	double *unext, *vnext, *pnext;
	double *temp;
	cudaMalloc(&ud, size);
	cudaMalloc(&vd, size);
	cudaMalloc(&pd, size);
	cudaMalloc(&unext, size);
	cudaMalloc(&vnext, size);
	cudaMalloc(&pnext, size);
	cudaMalloc(&temp, size);

	printf("Start processing..\n");

	// call gpu
	dim3 grid(ngx,ngy);
	dim3 block(nb,nb);
	dim3 block2(32,32);

	// kernel<<< grid, block >>>( nt, nit, ud, vd, pd, unext, vnext, pnext, temp );
	void *args[] = {(void *)&nt, (void *)&nit, (void *)&ud, (void *)&vd, (void *)&pd, (void *)&unext, (void *)&vnext, (void *)&pnext, (void *)&temp};
	int res = cudaLaunchCooperativeKernel( (void*)kernel, grid, block, args );

	// copy
	cudaMemcpy(u, unext, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(v, vnext, size, cudaMemcpyDeviceToHost);

	printf("Done %d\n",res);

	// print u
	FILE * ufile;
	ufile = fopen("u.txt","w");
	for( int j=0; j<ny; j++ ) {
		for( int i=0; i<nx; i++ ) {
			fprintf(ufile, "%f,", u[j*nx+i] );
		}
		fprintf(ufile, "\n");
	}
	fclose(ufile);

	// print v
	FILE * vfile;
	vfile = fopen("v.txt","w");
	for( int j=0; j<ny; j++ ) {
		for( int i=0; i<nx; i++ ) {
			fprintf(vfile, "%f,", v[j*nx+i] );
		}
		fprintf(vfile, "\n");
	}
	fclose(vfile);

	cudaFree( ud );
	cudaFree( vd );
	free( u );
	free( v );

	return 0;
}