/*
Fathul Asrar Alfansuri
19M38053

2D Navier-Stokes code written in C++.
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
void copyMap( double** origin, double** dest, int lx, int ly ) {
	for( int j=0; j<ly; j++ ) {
		for( int i=0; i<lx; i++ ) {
			dest[j][i] = origin[j][i];
		}
	}
}

// Build up equations
void build_up_b( int i, int j ) {
	double term1 = ( u[j][i+1]-u[j][i-1] )/( 2*dx );
	double term2 = ( v[j+1][i]-v[j-1][i] )/( 2*dy );

	double temp1 = (term1+term2)/dt;
	double temp2 = term1*term1;
	double temp4 = term2*term2;

	double temp31 = ( u[j+1][i]-u[j-1][i] )/( 2*dy );
	double temp32 = ( v[j][i+1]-v[j][i-1] )/( 2*dx );
	double temp3 = 2*temp31*temp32;

	double temp5 = rho * (temp1 - temp2 - temp3 - temp4);

	b[j][i] = temp5;
}

// Pressure poisson
void pressure_poisson( int i, int j ) {
	double temp11 = ( p[j][i+1]+p[j][i-1] ) * dy*dy;
	double temp12 = ( p[j+1][i]+p[j-1][i] ) * dx*dx;
	double temp1 = temp11+temp12;

	double temp2 = 2 * ( dx*dx + dy*dy );

	double temp31 = dx*dx * dy*dy;
	double temp32 = 2 * ( dx*dx + dy*dy );
	double temp3 = temp31/temp32 * b[j][i];

	double temp4 = temp1/temp2 - temp3;

	pnext[j][i] = temp4;
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

void cavity_flow( int i, int j ) {
	// u momentum
	double temp1 = u[j][i];
	double temp2 = u[j][i] * dt/dx * ( u[j][i]-u[j][i-1] );
	double temp3 = v[j][i] * dt/dy * ( u[j][i]-u[j-1][i] );
	double temp4 = dt/( 2*rho*dx ) * ( p[j][i+1]-p[j][i-1] );

	double temp51 = dt/(dx*dx) * ( u[j][i+1] - 2*u[j][i] + u[j][i-1] );
	double temp52 = dt/(dy*dy) * ( u[j+1][i] - 2*u[j][i] + u[j-1][i] );
	double temp5 = nu * (temp51+temp52);

	double temp6 = temp1 - temp2 - temp3 - temp4 + temp5;
	unext[j][i] = temp6;

	// v momentum
	temp1 = v[j][i];
	temp2 = u[j][i] * dt/dx * ( v[j][i]-v[j][i-1] );
	temp3 = v[j][i] * dt/dy * ( v[j][i]-v[j-1][i] );
	temp4 = dt/( 2*rho*dy ) * ( p[j+1][i]-p[j-1][i] );

	temp51 = dt/(dx*dx) * ( v[j][i+1] - 2*v[j][i] + v[j][i-1] );
	temp52 = dt/(dy*dy) * ( v[j+1][i] - 2*v[j][i] + v[j-1][i] );
	temp5 = nu * (temp51+temp52);

	temp6 = temp1 - temp2 - temp3 - temp4 + temp5;
	vnext[j][i] = temp6;
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
		for( int j=1; j<ny-1; j++ ) {
			for( int i=1; i<nx-1; i++ ) {
				build_up_b( i, j );
			}
		}

		// sub-iteration
		for( int ctr2=0; ctr2<nit; ctr2++ ) {

			// pressure poisson
			for( int j=1; j<ny-1; j++ ) {
				for( int i=1; i<nx-1; i++ ) {
					pressure_poisson( i, j );
				}
			}
			// --synchronize--
			for( int j=0; j<ny; j++ ) {
				for( int i=0; i<nx; i++ ) {
					pressure_poisson_2( i, j );
				}
			}
			copyMap( pnext, p, nx, ny );
		}

		// cavity flow
		for( int j=0; j<ny; j++ ) {
			for( int i=0; i<nx; i++ ) {
				cavity_flow_pre( i, j );
			}
		}
		// --synchronize--
		for( int j=1; j<ny-1; j++ ) {
			for( int i=1; i<nx-1; i++ ) {
				cavity_flow( i, j );
			}
		}
		copyMap( unext, u, nx, ny );
		copyMap( vnext, v, nx, ny );
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