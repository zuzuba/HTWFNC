#include <stdio.h>
#include <stdint.h>


// Struct to handle integer of 4 bits at a time
typedef struct {
   uint8_t i1 : 4;
   uint8_t i2 : 4;
} unit4x2_t;

void qmmm8bits( float lhs_scale, uint8_t lhs_offset, float rhs_scale,uint8_t rhs_offset, uint8_t* lhs_int_mat, uint8_t* rhs_int_mat, float result_scale, uint8_t result_offset, uint8_t* result_int_mat, int n,int k, int m ){

	int accumulator;

	for(int i=0;i<n;i++){
		for(int j=0;j<m;j++){
			for(int t=0; t<k;t++){
				accumulator = (lhs_int_mat[i*n + t] - lhs_offset) * (rhs_int_mat[t*m + j] - rhs_offset);
			}
		result_int_mat[i*n+j] = result_offset + (lhs_scale * rhs_scale/result_scale) * accumulator;
		}
	}
}


void qmm(float l_scale, float r_scale, float result_scale, unit4x2_t l_offset,  )