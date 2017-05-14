#include <stdio.h>
#include <stdint.h>
#include "utils.h"
#include "naive_qmm.h"
#include <math.h>


// Struct to handle integer of 4 bits at a time. ASSUMPTION: when we create a matrix i1 and i2 represent 2 consecutive entries
// typedef struct {
//    uint8_t i1 : 4;
//    uint8_t i2 : 4;
// } uint4x4_t;


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
void qmm_space_waste(float l_scale, float r_scale, float result_scale, uint4x1_t l_offset, uint4x1_t r_offset, 
	uint4x1_t result_offset, uint4x1_t* l_int_mat, uint4x1_t* r_int_mat, uint4x1_t* result_int_mat, 
	int n, int k, int m){

	int16_t accumulator;

	for(int i=0;i<n;i++){
		for(int j=0;j<m;j++){
			accumulator = 0;
			for(int t=0; t<k;t++){
				accumulator += (l_int_mat[i*k + t].i - l_offset.i) * (r_int_mat[t*m + j].i - r_offset.i);
			}
		result_int_mat[i*n+j].i = saturate(round(result_offset.i + (l_scale * r_scale/result_scale) * accumulator));
		}
	}
}


void qmm_naive(float l_scale, float r_scale, float result_scale, uint4x4_t l_offset, uint4x4_t r_offset, 
	uint4x4_t result_offset, uint4x4_t* l_int_mat, uint4x4_t* r_int_mat, uint4x4_t* result_int_mat, 
	int n, int k, int m){
	printf("Start naive\n");
	/*
	Every loop iteration makes us go through one uint4x4_t. The difference with the previous csae is that this
	contains 4 ints now and not one.
	*/
	// int16_t acc1, acc2, acc3, acc4;

	// for(int i=0; i<n/2; i = i+1){
	// 	for(int j=0; j<m/2; j = j+1){
	// 		acc1 = 0;
	// 		acc2 = 0;
	// 		acc3 = 0;
	// 		acc4 = 0;
	// 		printf("%d, %d\n", i, j);
	// 		for (int t=0; t<k/2; t = t+1){
	// 			acc1 += (l_int_mat[i*k + t].i1 - l_offset.i1) * (r_int_mat[t*m + j].i1 - r_offset.i1) + 
	// 					(l_int_mat[i*k + t].i2 - l_offset.i1) * (r_int_mat[t*m + j].i3 - r_offset.i1);

	// 			acc2 += (l_int_mat[i*k + t].i1 - l_offset.i1) * (r_int_mat[t*m + j].i2 - r_offset.i1) + 
	// 					(l_int_mat[i*k + t].i2 - l_offset.i1) * (r_int_mat[t*m + j].i4 - r_offset.i1);

	// 			acc3 += (l_int_mat[i*k + t].i3 - l_offset.i1) * (r_int_mat[t*m + j].i1 - r_offset.i1) + 
	// 					(l_int_mat[i*k + t].i4 - l_offset.i1) * (r_int_mat[t*m + j].i3 - r_offset.i1);

	// 			acc4 += (l_int_mat[i*k + t].i3 - l_offset.i1) * (r_int_mat[t*m + j].i2 - r_offset.i1) + 
	// 					(l_int_mat[i*k + t].i4 - l_offset.i1) * (r_int_mat[t*m + j].i4 - r_offset.i1);

	// 		}
	// 	result_int_mat[i*n + j].i1 = saturate(round(result_offset.i1 + (l_scale * r_scale/result_scale) * acc1));
	// 	result_int_mat[i*n + j].i2 = saturate(round(result_offset.i1 + (l_scale * r_scale/result_scale) * acc2));
	// 	result_int_mat[i*n + j].i3 = saturate(round(result_offset.i1 + (l_scale * r_scale/result_scale) * acc3));
	// 	result_int_mat[i*n + j].i4 = saturate(round(result_offset.i1 + (l_scale * r_scale/result_scale) * acc4));
		
	// 	}
	// }
	printf("Ciao\n");
}