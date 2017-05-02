#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <limits.h>
#include <stdlib.h>
#include <math.h>

double max(double *a, double *b){
	return *a < *b ? *b : *a;
}

double min(double *a, double *b){
	return *a < *b ? *a : *b;
}

void quantize(double **d, uint8_t **q, double *mn, double *mx, int n){
	*mx = INT_MIN, *mn = INT_MAX;

	// Find min and max of matrix
	for(int i = 0; i<n; i++){
		for(int j = 0; j<n; j++){
			*mx = max(mx,&d[i][j]);
			*mn = min(mn,&d[i][j]);
		}
	}

	// Compute size of linear cell
	double delta = (*mx - *mn)/255;

	// Determine uint8_t representation
	for(int i = 0; i<n; i++){
		for(int j = 0; j<n; j++)
			q[i][j] = (d[i][j] - *mn)/delta;
	}


}

void dequantize(double **d, uint8_t **q, double *mn, double *mx, int n){
	double delta = (*mx - *mn)/256; // size of linear cell
	double runner = 0.0;
	for(int i = 0; i<n; i++)
		for(int j = 0; j<n; j++)
			d[i][j] = *mn + q[i][j]*delta; // if q[i][j] == 0, then d[i][j] == mn
}

int main() {
	int n=3;
	double **d = (double **)malloc( n * sizeof(double *) );
	double **ded = (double **)malloc( n * sizeof(double *) );
	uint8_t **q = (uint8_t **)malloc( n*n * sizeof(uint8_t) );
	for (int i = 0; i<n; i++) {
		d[i] = (double *)malloc( n * sizeof(double) );
		ded[i] = (double *)malloc( n * sizeof(double) );
		q[i] = (uint8_t *)malloc( n * sizeof(uint8_t) );
	}

	for(int i = 0; i<n; i++){
		for(int j = 0; j<n; j++)
			d[i][j] = i*j+2;
	}
	double mn,mx;
	quantize(d,q,&mn,&mx,n);
	dequantize(ded,q,&mn,&mx,n);

	for(int i = 0; i<n; i++){
		for(int j = 0; j<n; j++){
			printf("%f ", d[i][j]);
		}
		printf("\n");
	}

	for(int i = 0; i<n; i++){
		for(int j = 0; j<n; j++){
			printf("%d ", q[i][j]);
		}
		printf("\n");
	}

	for(int i = 0; i<n; i++){
		for(int j = 0; j<n; j++){
			printf("%f ", ded[i][j]);
		}
		printf("\n");
	}



	//insert validation framework here...
}


