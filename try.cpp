#include<iostream>
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <limits.h>
#include <stdlib.h>
#include <math.h>
//#include "utils.h"
//#include "quantize.h"
#include <immintrin.h>

int test(){
	float qmin = 0.0, qmax = 15.0;
	__m256 qmin_avx = _mm256_broadcast_ss(&qmin);
	__m256 qmax_avx = _mm256_broadcast_ss(&qmax);
	__m256 upper_row;
	__m256 lower_row;
	__m256i upper_rowi;
	__m256i lower_rowi;
	__m256 top_shift = _mm256_set_ps(256.0,4096.0,256.0,4096.0,256.0,4096.0,256.0,4096.0);
	__m256 bottom_shift = _mm256_set_ps(1.0,16.0,1.0,16.0,1.0,16.0,1.0,16.0);
	float f[8] = {1.0,15.0,12.0,3.0,5.0,13.0,9.0,7.0};
	uint16_t fi[16];
	float g[8] = {2.0,14.0,11.0,4.0,6.0,1.0,10.0,8.0};
	uint16_t gi[16];
	upper_row = _mm256_loadu_ps(f);
        lower_row = _mm256_loadu_ps(g);
        //upper_row = _mm256_fmadd_ps(scale_avx,upper_row,zp_avx);
        //lower_row = _mm256_fmadd_ps(scale_avx,lower_row,zp_avx);
        upper_row = _mm256_max_ps( qmin_avx, _mm256_min_ps( qmax_avx, _mm256_round_ps(upper_row,_MM_FROUND_TO_NEAREST_INT)));
	upper_row = _mm256_cvttps_epi32( _mm256_mul_ps( top_shift, upper_row) );
	lower_row = _mm256_max_ps( qmin_avx, _mm256_min_ps( qmax_avx, _mm256_round_ps(lower_row,_MM_FROUND_TO_NEAREST_INT)));
	lower_row = _mm256_cvttps_epi32( _mm256_mul_ps( bottom_shift, lower_row) );
 
	//lower_row = _mm256_cvttps_epi32( bottom_shift, _mm256_cvttps_epi32( _mm256_max_ps( qmin_avx, _mm256_min_ps( qmax_avx, _mm256_round_ps(lower_row,_MM_FROUND_TO_NEAREST_INT)))));
        upper_row = _mm256_or_si256( upper_row,lower_row);
	
	//upper_rowi = _mm256_or_si256(upper_rowi, _mm256_slli_si256( upper_rowi,32));
	_mm256_store_si256( (__m256i*)&fi, upper_row);//_mm256_or_si256(upper_rowi,upper_rowi));
	_mm256_store_si256( (__m256i*)&gi, lower_row);//_mm256_or_si256(upper_rowi,upper_rowi));
	
	bool a;
	int b;
	for (int i = 0; i<16; i++){
		b = fi[i];
		for (int j = 16; j>0; j-- ){
			a = (b&(1<<(j-1))) > 0 ? 1 : 0;
			std::cout << a;
		}
		std::cout << " ";
	}
	for (int i = 0; i<16; i++){
		std::cout << fi[i];
		std::cout << " ";
	}
	std::cout << std::endl;
//	bool a;
	for (int i = 0; i<16; i++){
		for (int j = 16; j>0; j-- ){
			a = (gi[i]&(1<<(j-1))) > 0 ? 1 : 0;
			std::cout << a;
		}
		std::cout << " ";
	}
	for (int i = 0; i<16; i++){
		std::cout << gi[i];
		std::cout << " ";
	}

	return 0;
}

int main() {
	test();
}
