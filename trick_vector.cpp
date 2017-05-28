#include <stdio.h>
#include <stdint.h>
#include "utils.h"
#include "trick_vector.h"
#include <math.h>
#include <stdlib.h>
#include <immintrin.h> 

void inline uint4x4_to_mm256_row_shuffle_inline(uint4x4_t* a, __m256i *b1, __m256i *b2){

	__m256i tmp = _mm256_loadu_si256((__m256i const *)a);

	__m256i mask13 = _mm256_set1_epi8(15);
	__m256i odd = _mm256_and_si256(tmp, mask13);
	
	__m256i mask24 = _mm256_set1_epi8(240);
	__m256i even = _mm256_and_si256(tmp, mask24);
	

	__m256i blend_mask = _mm256_set1_epi16(32768);
	*b1 = _mm256_blendv_epi8 (odd, _mm256_slli_epi64 (even, 4), blend_mask);
	*b2 = _mm256_blendv_epi8 (_mm256_srli_epi64 (odd, 8), _mm256_srli_epi64 (even, 4), blend_mask);

}

void trick_vector_naive(uint4x4_t* l_int_mat, uint4x4_t* r_int_mat,uint4x4_t l_offset, uint4x4_t r_offset,int n, int k, int m,
	uint16_t* term2, uint16_t* term3, uint16_t* term4){

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

	*term4 = l_offset.i1*r_offset.i1*(k * 2);

}


void trick_vector_AVX(uint4x4_t* l_int_mat, uint4x4_t* r_int_mat,uint4x4_t l_offset, uint4x4_t r_offset,int n, int k, int m,
	uint16_t* term2, uint16_t* term3, uint16_t* term4){

	uint4x4_t r_elment,l_element;
	__m256i b1,b2,sum,offset,sum_1,sum_2;
	int i,j;
	uint16_t sum1, sum2;
	offset = _mm256_set1_epi16(l_offset.i1);

	// Sum over the entries of each col of rhs and multiply by left offset 
	for(j =0; j<m-15; j+=16){
		sum = _mm256_set1_epi16(0);
		for(i = 0; i<k; i++){
			uint4x4_to_mm256_row_shuffle_inline((r_int_mat + i*m + j),&b1,&b2); 
			sum = _mm256_add_epi16(sum,b1);
			sum = _mm256_add_epi16(sum,b2);

		}
		sum = _mm256_mullo_epi16(sum,offset);
		_mm256_storeu_si256( (__m256i *) term2,sum);	

		sum1 = 0;
		sum2 = 0;
		for(; i<k; i++){
			r_elment = r_int_mat[i*m + j]; 
			sum1 += r_elment.i1 + r_elment.i3;
			sum2 += r_elment.i2 + r_elment.i4;
		}
		term2[2*j] = l_offset.i1 * sum1;
		term2[2*j + 1] = l_offset.i1 * sum2;	
	}

	// Sum over the entries of each row of lhs and multiply by right offset
	for (i=0; i<n; i++){
		sum_1 = _mm256_set1_epi16(0);
		sum_2 = _mm256_set1_epi16(0);
		for (j=0; j<k-15; j+=16){
			uint4x4_to_mm256_row_shuffle_inline((r_int_mat + i*m + j),&b1,&b2);
			sum_1 = _mm256_add_epi16(sum_1,b1);
			sum_2 = _mm256_add_epi16(sum_2,b2);
		}
		sum1 = _mm256_haddsi_epi16(sum_1);
		sum2 = _mm256_haddsi_epi16(sum_2);
		for (; j<k; j++){
			l_element = l_int_mat[i*k + j];
			sum1 += l_element.i1 + l_element.i2;
			sum2 += l_element.i3 + l_element.i4;
		}
		term3[2*i] = r_offset.i1 * sum1;
		term3[2*i + 1] = r_offset.i1 * sum2;
	}

	*term4 = l_offset.i1*r_offset.i1*(k * 2);

}