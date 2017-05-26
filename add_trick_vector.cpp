#include <stdio.h>
#include <stdint.h>
#include "utils.h"
#include "add_trick_vector.h"
#include <math.h>
#include <stdlib.h>
#include <immintrin.h>

void add_trick_vector_naive(int16_t* acc, uint16_t* term2, uint16_t* term3, uint16_t term4, int n, int m){
	for(int i=0; i<n; i = i+1){
		for(int j=0; j<m; j = j+1){
			acc[2*i*2*m + 2*j]       += - term2[2*j]     - term3[2*i]     + term4;
			acc[2*i*2*m + 2*j+1]     += - term2[2*j + 1] - term3[2*i]     + term4;
			acc[(2*i+1)*2*m + 2*j]   += - term2[2*j]     - term3[2*i + 1] + term4;
			acc[(2*i+1)*2*m + 2*j+1] += - term2[2*j + 1] - term3[2*i + 1] + term4;
		}
	}
}
