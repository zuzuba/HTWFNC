#include <stdio.h>
#include <stdint.h>
#include "utils.h"
#include "qmm.h"
#include "qmm_kernel.h"
#include "trick_vector.h"
#include "round_saturation.h"
#include "add_trick_vector.h"
#include <math.h>
#include <stdlib.h>
#include <immintrin.h> 


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
		result_int_mat[i*m+j].i = saturate(round(result_offset.i + (l_scale * r_scale/result_scale) * accumulator));
		}
	}
}


void qmm_naive(float l_scale, float r_scale, float result_scale, uint4x4_t l_offset, uint4x4_t r_offset, 
	uint4x4_t result_offset, uint4x4_t* l_int_mat, uint4x4_t* r_int_mat, uint4x4_t* result_int_mat, 
	int n, int k, int m){
	/*
	Every loop iteration makes us go through one uint4x4_t. The difference with the previous csae is that this
	contains 4 ints now and not one.
	*/
	int16_t* acc = (int16_t*)malloc(sizeof(int16_t)*n*m);
	for (int i = 0; i < n*m; ++i) acc[i]=0;

	// Internally we divide everything by 2 because one uint4x4_t contains 4 integers of 4 bit organize in 2 cols and 2 rows
	n = n/2;
	m = m/2;
	k = k/2;

	qmm_kernel_naive(l_int_mat,r_int_mat,l_offset,r_offset,acc,n,k,m);
	round_saturation_naive(acc,result_int_mat,l_scale,r_scale,result_scale,result_offset,n,m);

}



void qmm_trick(float l_scale, float r_scale, float result_scale, uint4x4_t l_offset, uint4x4_t r_offset, 
	uint4x4_t result_offset, uint4x4_t* l_int_mat, uint4x4_t* r_int_mat, uint4x4_t* result_int_mat, 
	int n, int k, int m){

	uint16_t* term2 = (uint16_t*)malloc(sizeof(uint16_t)*m);
	uint16_t* term3 = (uint16_t*)malloc(sizeof(uint16_t)*n);
	uint16_t term4;
	int16_t* acc = (int16_t*)malloc(sizeof(int16_t)*n*m);
	for (int i = 0; i < n*m; ++i) acc[i]=0;

	// Internally we divide everything by 2 because one uint4x4_t contains 4 integers of 4 bit organize in 2 cols and 2 rows
	n = n/2;
	m = m/2;
	k = k/2;

	trick_vector_naive(l_int_mat,r_int_mat,l_offset,r_offset,n,k,m,term2,term3,&term4);
	qmm_kernel_trick(l_int_mat,r_int_mat,acc,n,k,m);
	add_trick_vector_naive(acc,term2,term3,term4,n,m);
	round_saturation_naive(acc,result_int_mat,l_scale,r_scale,result_scale,result_offset,n,m);

}



void qmm_trick_blocking(float l_scale, float r_scale, float result_scale, uint4x4_t l_offset, uint4x4_t r_offset, 
	uint4x4_t result_offset, uint4x4_t* l_int_mat, uint4x4_t* r_int_mat, uint4x4_t* result_int_mat, 
	int n, int k, int m){

	uint16_t* term2 = (uint16_t*)malloc(sizeof(uint16_t)*m);
	uint16_t* term3 = (uint16_t*)malloc(sizeof(uint16_t)*n);
	uint16_t term4;
	int16_t* acc = (int16_t*)malloc(sizeof(int16_t)*n*m);
	for (int i = 0; i < n*m; ++i) acc[i]=0;

	// Internally we divide everything by 2 because one uint4x4_t contains 4 integers of 4 bit organize in 2 cols and 2 rows
	n = n/2;
	m = m/2;
	k = k/2;

	trick_vector_naive(l_int_mat,r_int_mat,l_offset,r_offset,n,k,m,term2,term3,&term4);
	qmm_kernel_trick_blocking(l_int_mat,r_int_mat,acc,n,k,m);
	add_trick_vector_naive(acc,term2,term3,term4,n,m);
	round_saturation_naive(acc,result_int_mat,l_scale,r_scale,result_scale,result_offset,n,m);
}



void qmm_trick_AVX(float l_scale, float r_scale, float result_scale, uint4x4_t l_offset, uint4x4_t r_offset, 
	uint4x4_t result_offset, uint4x4_t* l_int_mat, uint4x4_t* r_int_mat, uint4x4_t* result_int_mat, 
	int n, int k, int m){

	uint16_t* term2 = (uint16_t*)malloc(sizeof(uint16_t)*m);
	uint16_t* term3 = (uint16_t*)malloc(sizeof(uint16_t)*n);
	uint16_t term4;
	int16_t* acc = (int16_t*)malloc(sizeof(int16_t)*n*m);
	for (int i = 0; i < n*m; ++i) acc[i]=0;

	// Internally we divide everything by 2 because one uint4x4_t contains 4 integers of 4 bit organize in 2 cols and 2 rows
	n = n/2;
	m = m/2;
	k = k/2;
	
	trick_vector_naive(l_int_mat,r_int_mat,l_offset,r_offset,n,k,m,term2,term3,&term4);
	qmm_kernel_trick_AVX(l_int_mat, r_int_mat,acc,n,k,m);
	add_trick_vector_naive(acc,term2,term3,term4,n,m);
	round_saturation_naive(acc,result_int_mat,l_scale,r_scale,result_scale,result_offset,n,m);
}