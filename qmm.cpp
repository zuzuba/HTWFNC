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
	uint4x4_t r_elment,l_element;
	uint16_t sum1, sum2;

	// Sum over the entries of each col of rhs and multiply by left offset 
	for(int j =0; j<m; j+=1){
		sum1 = 0;
		sum2 = 0;
		for(int i = 0; i<k; i++){
			r_elment = r_int_mat[i*m + j]; 
			sum1 += r_elment.i1 + r_elment.i3;
			sum2 += r_elment.i2 + r_elment.i4;
		}
		term2[2*j] = l_offset.i1 * sum1;
		term2[2*j + 1] = l_offset.i1 * sum2;	
	}

	// Sum over the entries of each row of lhs and multiply by right offset
	for (int i=0; i<n; i++){
		sum1 = 0;
		sum2 = 0;
		for (int j=0; j<k; j++){
			l_element = l_int_mat[i*k + j];
			sum1 += l_element.i1 + l_element.i2;
			sum2 += l_element.i3 + l_element.i4;
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
			for (int t=0; t<k; t = t+1){
				l_element = l_int_mat[i*k + t]; 
				r_elment = r_int_mat[t*m + j];
				acc1 += (l_element.i1) * (r_elment.i1) + 
						(l_element.i2) * (r_elment.i3);

				acc2 += (l_element.i1) * (r_elment.i2) + 
						(l_element.i2) * (r_elment.i4);

				acc3 += (l_element.i3) * (r_elment.i1) + 
						(l_element.i4) * (r_elment.i3);

				acc4 += (l_element.i3) * (r_elment.i2) + 
						(l_element.i4) * (r_elment.i4);

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

	int16_t acc11_int[16],acc12_int[16],acc21_int[16],acc22_int[16];
	int16_t acc1,acc2,acc3,acc4;
	uint4x4_t r_elment,l_element;
	// Internally we divide everything by 2 because one uint4x4_t contains 4 integers of 4 bit organize in 2 cols and 2 rows

	n = n/2;
	m = m/2;
	k = k/2;
	int t,i,j;

	// Precompute additional terms
	uint16_t* term2 = (uint16_t*)malloc(sizeof(uint16_t)* m *2);
	uint16_t* term3 = (uint16_t*)malloc(sizeof(uint16_t)* n *2);
	uint16_t term4;
	float scale = l_scale * r_scale/result_scale;
	uint16_t sum1, sum2;
	__m256i r1,r2;
	__m256i temp[32],temp_t[32];
	__m256i acc11[16],acc12[16],acc21[16],acc22[16],dot_prod1[32],dot_prod2[32];
	
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

	for( i=0; i<n; i = i+1){
		for( j=0; j<(m-15); j = j+16){
			
			for (int u = 0; u < 16; u++)
			{
				acc11_int[u] = 0;
				acc12_int[u] = 0;
				acc21_int[u] = 0;
				acc22_int[u] = 0;
				acc11[u] = _mm256_set1_epi16(0);
				acc12[u] = _mm256_set1_epi16(0);
				acc21[u] = _mm256_set1_epi16(0);
				acc22[u] = _mm256_set1_epi16(0);	
			}

			for ( t=0; t<(k-31); t = t+32){
				uint4x4_to_mm256_row_shuffle(l_int_mat + i*k + t, &r1, &r2);
				
				for (int u = 0; u < 16; u++)
				{
					uint4x4_to_mm256_row_shuffle(r_int_mat + (t+u)*m + j, &temp[2*u], &temp[2*u+1]);	
				}
				
				transpose(temp,temp_t);

				for (int u = 0; u < 32; u++)
				{
					dot_prod1[u] = _mm256_maddubs_epi16 (r1,temp_t[u]);
					dot_prod2[u] = _mm256_maddubs_epi16 (r2,temp_t[u]);
				}

				for (int u = 0; u < 16; u++)
				{
					acc11[u] = _mm256_add_epi16(acc11[u],dot_prod1[2*u]);
					acc12[u] = _mm256_add_epi16(acc12[u],dot_prod1[2*u+1]);
					acc21[u] = _mm256_add_epi16(acc21[u],dot_prod2[2*u]);
					acc22[u] = _mm256_add_epi16(acc22[u],dot_prod2[2*u+1]);
				}
				
			}

			for (int u = 0; u < 16; u++){
				acc11_int[u] += _mm256_haddsi_epi16(acc11[u]);
				acc12_int[u] += _mm256_haddsi_epi16(acc12[u]);
				acc21_int[u] += _mm256_haddsi_epi16(acc21[u]);
				acc22_int[u] += _mm256_haddsi_epi16(acc22[u]);
			}

			for (int t = 0; t < 16; t++)
			{
				for(int u=t;u<k; u = u+1){
					acc11_int[t] += (l_int_mat[i*k + u].i1) * (r_int_mat[u*m + j+t].i1) + 
							(l_int_mat[i*k + u].i2) * (r_int_mat[u*m + j].i3);

					acc12_int[t] += (l_int_mat[i*k + u].i1) * (r_int_mat[u*m + j+t].i2) + 
							(l_int_mat[i*k + u].i2) * (r_int_mat[u*m + j+t].i4);

					acc21_int[t] += (l_int_mat[i*k + u].i3) * (r_int_mat[u*m + j+t].i1) + 
							(l_int_mat[i*k + u].i4) * (r_int_mat[u*m + j].i3);

					acc22_int[t] += (l_int_mat[i*k + u].i3) * (r_int_mat[u*m + j+t].i2) + 
							(l_int_mat[i*k + u].i4) * (r_int_mat[u*m + j+t].i4);

				}
			}

		for (int u = 0; u < 16; u++)
		{
		acc11_int[u] = acc11_int[u] - term2[2*j] - term3[2*i] + term4;
		acc12_int[u] = acc12_int[u] - term2[2*j + 1] - term3[2*i] + term4;
		acc21_int[u] = acc21_int[u] - term2[2*j] - term3[2*i + 1] + term4;
		acc22_int[u] = acc22_int[u] - term2[2*j + 1] - term3[2*i + 1] + term4;

		result_int_mat[i*m + j+u].i1 = saturate(round(result_offset.i1 + scale * acc11_int[u]));
		result_int_mat[i*m + j+u].i2 = saturate(round(result_offset.i1 + scale * acc12_int[u]));
		result_int_mat[i*m + j+u].i3 = saturate(round(result_offset.i1 + scale * acc21_int[u]));
		result_int_mat[i*m + j+u].i4 = saturate(round(result_offset.i1 + scale * acc22_int[u]));
			
		}
		}

		for(; j<m; j = j+1){
			acc1 = 0;
			acc2 = 0;
			acc3 = 0;
			acc4 = 0;
			for (int t=0; t<k; t = t+1){
				l_element = l_int_mat[i*k + t]; 
				r_elment = r_int_mat[t*m + j];
				acc1 += (l_element.i1) * (r_elment.i1) + 
						(l_element.i2) * (r_elment.i3);

				acc2 += (l_element.i1) * (r_elment.i2) + 
						(l_element.i2) * (r_elment.i4);

				acc3 += (l_element.i3) * (r_elment.i1) + 
						(l_element.i4) * (r_elment.i3);

				acc4 += (l_element.i3) * (r_elment.i2) + 
						(l_element.i4) * (r_elment.i4);

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
