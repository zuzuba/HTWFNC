#include<stdio.h>
#include<time.h>
#include<stdint.h>
#include<limits.h>
#include<stdlib.h>
double max(double *a, double *b){
	return *a < *b ? *b : *a;
}

double min(double *a, double *b){
	return *a < *b ? *a : *b;
}

void quantize(double **d, uint8_t **q, double *mn, double *mx, int n){
	*mx = INT_MIN, *mn = INT_MAX;

	// Find min and max of matrix
	for(int i = 0; i<n; i++)
		for(int j = 0; j<n; j++){
			*mx = max(mx,&d[i][j]);
			*mn = min(mn,&d[i][j]);
		}

	// Compute size of linear cell
	double delta = (*mx - *mn)/256;
	uint8_t runner = 0;
	double current = 0.0;

	// Determine uint8_t representation
	for(int i = 0; i<n; i++)
		for(int j = 0; j<n; j++){
			runner = 0;
			current = d[i][j];
			while(runner < current) {
				runner += delta;
			}
			q[i][j] = runner;
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
	int n;
	scanf("%d",&n);
	double **d = (double **)malloc( n * sizeof(double) );
	uint8_t **q = (uint8_t **)malloc( n * sizeof(uint8_t) );
	for (int i = 0; i<n; i++) {
		d[i] = (double *)malloc( n * sizeof(double) );
		q[i] = (uint8_t *)malloc( n * sizeof(uint8_t) );
	}

	//insert validation framework here...
}


