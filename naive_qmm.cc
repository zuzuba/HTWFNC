#include <stdio.h>
#include <stdint.h>


// Struct to handle integer of 4 bits at a time. ASSUMPTION: when we create a matrix i1 and i2 represent 2 consecutive entries
typedef struct {
   uint8_t i1 : 4;
   uint8_t i2 : 4;
} unit4x2_t;

typedef struct {
   uint8_t i : 4;
} unit4x1_t;

void qmmm8bits( float lhs_scale, uint8_t lhs_offset, float rhs_scale,uint8_t rhs_offset, uint8_t* lhs_int_mat, uint8_t* rhs_int_mat, float result_scale, uint8_t result_offset, uint8_t* result_int_mat, int n,int k, int m ){

	int accumulator;

	for(int i=0;i<n;i++){
		for(int j=0;j<m;j++){
			accumulator = 0;
			for(int t=0; t<k;t++){
				accumulator += (lhs_int_mat[i*n + t] - lhs_offset) * (rhs_int_mat[t*m + j] - rhs_offset);
			}
		result_int_mat[i*n+j] = result_offset + (lhs_scale * rhs_scale/result_scale) * accumulator;
		}
	}
}


// Implementing the MM with struct the require 8 bits but contain only bits of info
void qmm_space_waste(float l_scale, float r_scale, float result_scale, unit4x1_t l_offset, unit4x1_t r_offset, 
	unit4x1_t result_offset, unit4x1_t* l_int_mat, unit4x1_t* r_int_mat, unit4x1_t* result_int_mat, 
	int n, int k, int m){

	uint16_t accumulator;

	for(int i=0;i<n;i++){
		for(int j=0;j<m;j++){
			accumulator = 0;
			for(int t=0; t<k;t++){
				accumulator += (lhs_int_mat[i*n + t].i - lhs_offset.i) * (rhs_int_mat[t*m + j].i - rhs_offset.i);
			}
		result_int_mat[i*n+j].i = result_offset.i + (lhs_scale * rhs_scale/result_scale) * accumulator;
		}
	}
}


void qmm(float l_scale, float r_scale, float result_scale, unit4x2_t l_offset, unit4x2_t r_offset, 
	unit4x2_t result_offset, unit4x2_t* l_int_mat, unit4x2_t* r_int_mat, unit4x2_t* result_int_mat, 
	int n, int k, int m){

	/*
	Every loop iteration makes us go through one uint4x2_t. The difference with the previous csae is that this
	contains 2ints now and not one.
	*/
	uint16_t acc1, acc2;

	for(int i=0;i<n;i++){
		for(int j=0;j<m;j++){
			acc1 = 0;
			acc2 = 0;
			for (int t=0; t<k; t = t+2){
				acc1 += (l_int_mat[i*n + t].i1 - l_offset.i1) * (r_int_mat[t*m + j].i1 - r_offset.i1) + 
				(l_int_mat[i*n + t].i2 - l_offset.i1) * (r_int_mat[t*m + 1 + j].i1 - r_offset.i1);

				acc2 += (l_int_mat[i*n + t].i1 - l_offset.i1) * (r_int_mat[t*m + j].i2 - r_offset.i1) + 
				(l_int_mat[i*n + t].i2 - l_offset.i1) * (r_int_mat[t*m + 1 + j].i2 - r_offset.i1);

			}
		result_int_mat[i*n + j].i1 = result_offset.i1 + (lhs_scale * rhs_scale/result_scale) * acc1;

		result_int_mat[i*n + j].i2 = result_offset.i1 + (lhs_scale * rhs_scale/result_scale) * acc2;
		
		}
	}
}