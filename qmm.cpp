#include <stdio.h>
#include <stdint.h>
#include "utils.h"
#include "qmm.h"
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
	int16_t acc1, acc2, acc3, acc4;

	// Internally we divide everything by 2 because one uint4x4_t contains 4 integers of 4 bit organize in 2 cols and 2 rows

	n = n/2;
	m = m/2;
	k = k/2;

	for(int i=0; i<n; i = i+1){
		for(int j=0; j<m; j = j+1){
			acc1 = 0;
			acc2 = 0;
			acc3 = 0;
			acc4 = 0;
			for (int t=0; t<k; t = t+1){
				acc1 += (l_int_mat[i*k + t].i1 - l_offset.i1) * (r_int_mat[t*m + j].i1 - r_offset.i1) + 
						(l_int_mat[i*k + t].i2 - l_offset.i1) * (r_int_mat[t*m + j].i3 - r_offset.i1);

				acc2 += (l_int_mat[i*k + t].i1 - l_offset.i1) * (r_int_mat[t*m + j].i2 - r_offset.i1) + 
						(l_int_mat[i*k + t].i2 - l_offset.i1) * (r_int_mat[t*m + j].i4 - r_offset.i1);

				acc3 += (l_int_mat[i*k + t].i3 - l_offset.i1) * (r_int_mat[t*m + j].i1 - r_offset.i1) + 
						(l_int_mat[i*k + t].i4 - l_offset.i1) * (r_int_mat[t*m + j].i3 - r_offset.i1);

				acc4 += (l_int_mat[i*k + t].i3 - l_offset.i1) * (r_int_mat[t*m + j].i2 - r_offset.i1) + 
						(l_int_mat[i*k + t].i4 - l_offset.i1) * (r_int_mat[t*m + j].i4 - r_offset.i1);

			}
			
		result_int_mat[i*m + j].i1 = saturate(round(result_offset.i1 + (l_scale * r_scale/result_scale) * acc1));
		result_int_mat[i*m + j].i2 = saturate(round(result_offset.i1 + (l_scale * r_scale/result_scale) * acc2));
		result_int_mat[i*m + j].i3 = saturate(round(result_offset.i1 + (l_scale * r_scale/result_scale) * acc3));
		result_int_mat[i*m + j].i4 = saturate(round(result_offset.i1 + (l_scale * r_scale/result_scale) * acc4));
		
		}
	}
}



void qmm_trick(float l_scale, float r_scale, float result_scale, uint4x4_t l_offset, uint4x4_t r_offset, 
	uint4x4_t result_offset, uint4x4_t* l_int_mat, uint4x4_t* r_int_mat, uint4x4_t* result_int_mat, 
	int n, int k, int m){

	int16_t acc1, acc2, acc3, acc4;

	// Internally we divide everything by 2 because one uint4x4_t contains 4 integers of 4 bit organize in 2 cols and 2 rows

	n = n/2;
	m = m/2;
	k = k/2;


	// Precompute additional terms
	uint16_t* term2 = (uint16_t*)malloc(sizeof(uint16_t)* m *2);
	uint16_t* term3 = (uint16_t*)malloc(sizeof(uint16_t)* n *2);
	uint16_t term4;

	uint16_t sum1, sum2;
	
	// Sum over the entries of each col of rhs and multiply by left offset 
	for(int j =0; j<m; j++){
		sum1 = 0;
		sum2 = 0;
		for(int i = 0; i<k; i++){
			sum1 += r_int_mat[i*m + j].i1 + r_int_mat[i*m + j].i3;
			sum2 += r_int_mat[i*m + j].i2 + r_int_mat[i*m + j].i4;
		}
		term2[2*j] = l_offset.i1 * sum1;
		term2[2*j + 1] = l_offset.i1 * sum2;	
	}

	// Sum over the entries of each row of lhs and multiply by rigt offset
	for (int i=0; i<n; i++){
		sum1 = 0;
		sum2 = 0;
		for (int j=0; j<k; j++){
			sum1 += l_int_mat[i*k + j].i1 + l_int_mat[i*k + j].i2;
			sum2 += l_int_mat[i*k + j].i3 + l_int_mat[i*k + j].i4;
		}
		term3[2*i] = r_offset.i1 * sum1;
		term3[2*i + 1] = r_offset.i1 * sum2;
	}

	term4 = l_offset.i1 * r_offset.i1 * (k * 2);
	
	// It should be correct up until here

	for(int i=0; i<n; i = i+1){
		for(int j=0; j<m; j = j+1){
			acc1 = 0;
			acc2 = 0;
			acc3 = 0;
			acc4 = 0;
			for (int t=0; t<k; t = t+1){
				acc1 += (l_int_mat[i*k + t].i1) * (r_int_mat[t*m + j].i1) + 
						(l_int_mat[i*k + t].i2) * (r_int_mat[t*m + j].i3);

				acc2 += (l_int_mat[i*k + t].i1) * (r_int_mat[t*m + j].i2) + 
						(l_int_mat[i*k + t].i2) * (r_int_mat[t*m + j].i4);

				acc3 += (l_int_mat[i*k + t].i3) * (r_int_mat[t*m + j].i1) + 
						(l_int_mat[i*k + t].i4) * (r_int_mat[t*m + j].i3);

				acc4 += (l_int_mat[i*k + t].i3) * (r_int_mat[t*m + j].i2) + 
						(l_int_mat[i*k + t].i4) * (r_int_mat[t*m + j].i4);

			}
			
		acc1 = acc1 - term2[2*j] - term3[2*i] + term4;
		acc2 = acc2 - term2[2*j + 1] - term3[2*i] + term4;
		acc3 = acc3 - term2[2*j] - term3[2*i + 1] + term4;
		acc4 = acc4 - term2[2*j + 1] - term3[2*i + 1] + term4;

		result_int_mat[i*m + j].i1 = saturate(round(result_offset.i1 + (l_scale * r_scale/result_scale) * acc1));
		result_int_mat[i*m + j].i2 = saturate(round(result_offset.i1 + (l_scale * r_scale/result_scale) * acc2));
		result_int_mat[i*m + j].i3 = saturate(round(result_offset.i1 + (l_scale * r_scale/result_scale) * acc3));
		result_int_mat[i*m + j].i4 = saturate(round(result_offset.i1 + (l_scale * r_scale/result_scale) * acc4));
		
		}
	}

}


void qmm_trick_AVX(float l_scale, float r_scale, float result_scale, uint4x4_t l_offset, uint4x4_t r_offset, 
	uint4x4_t result_offset, uint4x4_t* l_int_mat, uint4x4_t* r_int_mat, uint4x4_t* result_int_mat, 
	int n, int k, int m){

	int16_t acc1, acc2, acc3, acc4,acc1_, acc2_, acc3_, acc4_;

	// Internally we divide everything by 2 because one uint4x4_t contains 4 integers of 4 bit organize in 2 cols and 2 rows

	n = n/2;
	m = m/2;
	k = k/2;
	int t;

	// Precompute additional terms
	uint16_t* term2 = (uint16_t*)malloc(sizeof(uint16_t)* m *2);
	uint16_t* term3 = (uint16_t*)malloc(sizeof(uint16_t)* n *2);
	uint16_t term4;

	uint16_t sum1, sum2;
	
	// Sum over the entries of each col of rhs and multiply by left offset 
	for(int j =0; j<m; j++){
		sum1 = 0;
		sum2 = 0;
		for(int i = 0; i<k; i++){
			sum1 += r_int_mat[i*m + j].i1 + r_int_mat[i*m + j].i3;
			sum2 += r_int_mat[i*m + j].i2 + r_int_mat[i*m + j].i4;
		}
		term2[2*j] = l_offset.i1 * sum1;
		term2[2*j + 1] = l_offset.i1 * sum2;	
	}

	// Sum over the entries of each row of lhs and multiply by rigt offset
	for (int i=0; i<n; i++){
		sum1 = 0;
		sum2 = 0;
		for (int j=0; j<k; j++){
			sum1 += l_int_mat[i*k + j].i1 + l_int_mat[i*k + j].i2;
			sum2 += l_int_mat[i*k + j].i3 + l_int_mat[i*k + j].i4;
		}
		term3[2*i] = r_offset.i1 * sum1;
		term3[2*i + 1] = r_offset.i1 * sum2;
	}

	term4 = l_offset.i1 * r_offset.i1 * (k * 2);

	for(int i=0; i<n; i = i+1){
		for(int j=0; j<m; j = j+1){
			acc1 = 0;
			acc2 = 0;
			acc3 = 0;
			acc4 = 0;
			 __m256i b1,b2,b3,b4;
			 uint4x4_t column[16];
			for ( t=0; t<(k-15); t = t+16){
				uint4x4_to_mm256_row(l_int_mat + i*k + t, &b1, &b2);
				for (int u = 0; u < 16; u++)
				{
					column[u].i1 = r_int_mat[(t+u)*m + j].i1;
					column[u].i2 = r_int_mat[(t+u)*m + j].i2;
					column[u].i3 = r_int_mat[(t+u)*m + j].i3;
					column[u].i4 = r_int_mat[(t+u)*m + j].i4;
				}
				uint4x4_to_mm256_column(column, &b3, &b4);

				acc1 += dot_prod_AVX(b1,b3);

				acc2 += dot_prod_AVX(b1,b4);

				acc3 += dot_prod_AVX(b2,b3);

				acc4 += dot_prod_AVX(b2,b4);

			}

			for(int u=t;u<k; u = u+1){
				acc1 += (l_int_mat[i*k + u].i1) * (r_int_mat[u*m + j].i1) + 
						(l_int_mat[i*k + u].i2) * (r_int_mat[u*m + j].i3);

				acc2 += (l_int_mat[i*k + u].i1) * (r_int_mat[u*m + j].i2) + 
						(l_int_mat[i*k + u].i2) * (r_int_mat[u*m + j].i4);

				acc3 += (l_int_mat[i*k + u].i3) * (r_int_mat[u*m + j].i1) + 
						(l_int_mat[i*k + u].i4) * (r_int_mat[u*m + j].i3);

				acc4 += (l_int_mat[i*k + u].i3) * (r_int_mat[u*m + j].i2) + 
						(l_int_mat[i*k + u].i4) * (r_int_mat[u*m + j].i4);

			}

			
		acc1 = acc1 - term2[2*j] - term3[2*i] + term4;
		acc2 = acc2 - term2[2*j + 1] - term3[2*i] + term4;
		acc3 = acc3 - term2[2*j] - term3[2*i + 1] + term4;
		acc4 = acc4 - term2[2*j + 1] - term3[2*i + 1] + term4;

		result_int_mat[i*m + j].i1 = saturate(round(result_offset.i1 + (l_scale * r_scale/result_scale) * acc1));
		result_int_mat[i*m + j].i2 = saturate(round(result_offset.i1 + (l_scale * r_scale/result_scale) * acc2));
		result_int_mat[i*m + j].i3 = saturate(round(result_offset.i1 + (l_scale * r_scale/result_scale) * acc3));
		result_int_mat[i*m + j].i4 = saturate(round(result_offset.i1 + (l_scale * r_scale/result_scale) * acc4));
		
		}
	}

}
