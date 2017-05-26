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

void add_trick_vector_AVX(int16_t* acc, uint16_t* term2, uint16_t* term3, uint16_t term4, int n, int m){

	__m256i term4_m256 =  _mm256_set1_epi16(term4);
	__m256i term3_m256;
	__m256i term2_m256;
	__m256i acc_m256;
	__m256i t;
	int j;
	uint16_t term3_;

	for (int i = 0; i < 2*n; i++)
	{
		term3_ = term3[i];
		term3_m256 = _mm256_set1_epi16(term3_);
		for (j = 0; i < m-15; j+=16)
		{
			term2_m256 = _mm256_loadu_si256((__m256i const *) (term2 + 2*j));
			acc_m256 = _mm256_loadu_si256((__m256i *) (acc + i*2*m + 2*j) );
			t = _mm256_add_epi16 (term2_m256,term3_m256);
			t = _mm256_sub_epi16 (term4_m256,t);
			t = _mm256_add_epi16 (t, acc_m256);
			_mm256_storeu_si256 ((__m256i *) acc + i*2*m + 2*j, t);
		}

		for (; j < m; ++j)
		{
			acc[i*2*m + 2*j]       += - term2[2*j]     - term3_     + term4;
			acc[i*2*m + 2*j+1]     += - term2[2*j + 1] - term3_     + term4;
		}
	}

}