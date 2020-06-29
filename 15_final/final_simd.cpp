/*
Fathul Asrar Alfansuri
19M38053

2D Navier-Stokes code written in C++, parallelized using both OpenMP and SIMD.
The result of this code is 2 files, 'u.txt' and 'v.txt', a comma-separated data file.
*/

#include <stdio.h>
#include <cstdio>
#include <cstdlib>

using namespace std;

// consts
int nt = 500;
int nit = 50;
const int nx = 41;
const int ny = 41;
const int c = 1;
const double dx = 2.0 / (nx - 1);
const double dy = 2.0 / (ny - 1);

const double rho = 1;
const double nu = 0.1;
const double dt = 0.001;

// maps
double **u, **v, **p, **b;
double **unext, **vnext, **pnext;

// Init maps
void initMap() {
	u = new double*[ny];
	v = new double*[ny];
	p = new double*[ny];
	b = new double*[ny];
	unext = new double*[ny];
	vnext = new double*[ny];
	pnext = new double*[ny];

	for( int i=0; i<ny; i++ ) {
		u[i] = new double[nx];
		v[i] = new double[nx];
		p[i] = new double[nx];
		b[i] = new double[nx];
		unext[i] = new double[nx];
		vnext[i] = new double[nx];
		pnext[i] = new double[nx];

		for( int j=0; j<nx; j++ ) {
			u[i][j] = 0;
			v[i][j] = 0;
			p[i][j] = 0;
			b[i][j] = 0;
			unext[i][j] = 0;
			vnext[i][j] = 0;
			pnext[i][j] = 0;
		}
	}
}

// copy buffers
void copyMap( double** origin, double** dest, int j ) {
	for( int i=0; i<nx; i++ ) {
		dest[j][i] = origin[j][i];
	}
}

// Build up equations
void build_up_b( int j ) {
	double *term1 = new double[nx];
	double *term2 = new double[nx];
	double *temp1 = new double[nx];
	double *temp2 = new double[nx];
	double *temp3 = new double[nx];
	double *temp31 = new double[nx];
	double *temp32 = new double[nx];
	double *temp4 = new double[nx];
	double *temp5 = new double[nx];

	for( int i=1; i<nx-1; i++ ) {
		term1[i] = ( u[j][i+1]-u[j][i-1] )/( 2*dx );
		term2[i] = ( v[j+1][i]-v[j-1][i] )/( 2*dy );
		temp1[i] = (term1[i]+term2[i])/dt;
		temp2[i] = term1[i]*term1[i];
		temp4[i] = term2[i]*term2[i];
		temp31[i] = ( u[j+1][i]-u[j-1][i] )/( 2*dy );
		temp32[i] = ( v[j][i+1]-v[j][i-1] )/( 2*dx );
		temp3[i] = 2*temp31[i]*temp32[i];
		b[j][i] = rho * (temp1[i] - temp2[i] - temp3[i] - temp4[i]);
	}
}

// Pressure poisson
void pressure_poisson( int j ) {
	double *temp11 = new double[nx];
	double *temp12 = new double[nx];
	double *temp1 = new double[nx];
	double temp2;
	double temp31;
	double temp32;
	double *temp3 = new double[nx];

	for( int i=1; i<nx-1; i++ ) {
		temp11[i] = ( p[j][i+1]+p[j][i-1] ) * dy*dy;
		temp12[i] = ( p[j+1][i]+p[j-1][i] ) * dx*dx;
		temp1[i] = temp11[i]+temp12[i];
		temp2 = 2 * ( dx*dx + dy*dy );
		temp31 = dx*dx * dy*dy;
		temp32 = 2 * ( dx*dx + dy*dy );
		temp3[i] = temp31/temp32 * b[j][i];
		pnext[j][i] = temp1[i]/temp2 - temp3[i];
	}
}
void pressure_poisson_2( int i, int j ) {
	// calculate edges after synchronized
	if (i==nx-1) {
		pnext[j][i] = pnext[j][i-1];
	}
	else if (i==0) {
		pnext[j][i] = pnext[j][i+1];
	}
	else if (j==0) {
		pnext[j][i] = pnext[j+1][i];
	}
	else if (j==ny-1) {
		pnext[j][i] = 0;
	}
}

void cavity_flow_pre( int i, int j ) {
	// calculate initial value (especially on edges)
	if (j==ny-1) {
		unext[j][i] = 1;
		vnext[j][i] = 0;
		return;
	}
	unext[j][i] = 0;
	vnext[j][i] = 0;
}

void cavity_flow( int j ) {
	double *temp1 = new double[nx];
	double *temp2 = new double[nx];
	double *temp3 = new double[nx];
	double *temp4 = new double[nx];
	double *temp51 = new double[nx];
	double *temp52 = new double[nx];
	double *temp5 = new double[nx];

	// u momentum
	for( int i=1; i<nx-1; i++ ) {
		temp1[i] = u[j][i];
		temp2[i] = u[j][i] * dt/dx * ( u[j][i]-u[j][i-1] );
		temp3[i] = v[j][i] * dt/dy * ( u[j][i]-u[j-1][i] );
		temp4[i] = dt/( 2*rho*dx ) * ( p[j][i+1]-p[j][i-1] );
		temp51[i] = dt/(dx*dx) * ( u[j][i+1] - 2*u[j][i] + u[j][i-1] );
		temp52[i] = dt/(dy*dy) * ( u[j+1][i] - 2*u[j][i] + u[j-1][i] );
		temp5[i] = nu * (temp51[i]+temp52[i]);
		unext[j][i] = temp1[i] - temp2[i] - temp3[i] - temp4[i] + temp5[i];
	}

	// v momentum
	for( int i=1; i<nx-1; i++ ) {
		temp1[i] = v[j][i];
		temp2[i] = u[j][i] * dt/dx * ( v[j][i]-v[j][i-1] );
		temp3[i] = v[j][i] * dt/dy * ( v[j][i]-v[j-1][i] );
		temp4[i] = dt/( 2*rho*dy ) * ( p[j+1][i]-p[j-1][i] );
		temp51[i] = dt/(dx*dx) * ( v[j][i+1] - 2*v[j][i] + v[j][i-1] );
		temp52[i] = dt/(dy*dy) * ( v[j+1][i] - 2*v[j][i] + v[j-1][i] );
		temp5[i] = nu * (temp51[i]+temp52[i]);
		vnext[j][i] = temp1[i] - temp2[i] - temp3[i] - temp4[i] + temp5[i];
	}
}

int main() {
	// inits
	initMap();
	nt = 300;
	nit = 50;

	printf("Start processing..\n");

	// main iteration
	for( int ctr1=0; ctr1<nt; ctr1++ ) {
	    printf("loop no. %d\n",ctr1);

		// build up b
		#pragma omp parallel for
		for( int j=1; j<ny-1; j++ ) {
			build_up_b( j );
		}

		// sub-iteration
		for( int ctr2=0; ctr2<nit; ctr2++ ) {

			// pressure poisson
			#pragma omp parallel for
			for( int j=1; j<ny-1; j++ ) {
				pressure_poisson( j );
			}
			// --synchronize--
			#pragma omp parallel for collapse(2)
			for( int j=0; j<ny; j++ ) {
				for( int i=0; i<nx; i++ ) {
					pressure_poisson_2( i, j );
				}
			}
			#pragma omp parallel for
			for( int j=0; j<ny; j++ ) {
				copyMap( pnext, p, j );
			}
		}

		// cavity flow pre
		#pragma omp parallel for collapse(2)
		for( int j=0; j<ny; j++ ) {
			for( int i=0; i<nx; i++ ) {
				cavity_flow_pre( i, j );
			}
		}
		// --synchronize--
		// cavity flow
		#pragma omp parallel for
		for( int j=1; j<ny-1; j++ ) {
			cavity_flow( j );
		}
		#pragma omp parallel for
		for( int j=0; j<ny; j++ ) {
			copyMap( unext, u, j );
			copyMap( vnext, v, j );
		}
	}
	printf("Done\n");

	// print u
	FILE * ufile;
	ufile = fopen("u.txt","w");
	for( int j=0; j<ny; j++ ) {
		for( int i=0; i<nx; i++ ) {
			fprintf(ufile, "%f,", u[j][i] );
		}
		fprintf(ufile, "\n");
	}
	fclose(ufile);

	// print v
	FILE * vfile;
	vfile = fopen("v.txt","w");
	for( int j=0; j<ny; j++ ) {
		for( int i=0; i<nx; i++ ) {
			fprintf(vfile, "%f,", v[j][i] );
		}
		fprintf(vfile, "\n");
	}
	fclose(vfile);

	return 0;
}