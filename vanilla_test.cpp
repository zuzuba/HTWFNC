//#include"vanilla_test.h"
#include <stdint.h>

void vanilla(double **d, uint8_t **q, double *mn, double *mx, int n){
	*mn = *mn/2;
	*mx = *mx * 2;
	for (int i = 0; i<n; i++){
		for (int j=0; j<n; j++){
			d[i][j] = d[i][j] + 1;
			q[i][j] = q[i][j] - 1;
		}
	}
}