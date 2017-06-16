#include <stdio.h>
#include <stdint.h>
#include "utils.h"
#include "round_saturation.h"
#include <math.h>
#include <stdlib.h>
#include <immintrin.h> 

void round_saturation_naive(int16_t* acc,uint4x4_t* result_int_mat,float l_scale, float r_scale, float result_scale,uint4x4_t result_offset, int n, int m){

float scale = l_scale * r_scale/result_scale;

	for(int i=0; i<n; i = i+1){
		for(int j=0; j<m; j = j+1){
			result_int_mat[i*m + j].i1 = saturate(round(result_offset.i1 + (scale) * acc[(2*i)*2*m + 2*j]));
			result_int_mat[i*m + j].i2 = saturate(round(result_offset.i1 + (scale) * acc[(2*i)*2*m + 2*j+1]));
			result_int_mat[i*m + j].i3 = saturate(round(result_offset.i1 + (scale) * acc[(2*i+1)*2*m + 2*j]));
			result_int_mat[i*m + j].i4 = saturate(round(result_offset.i1 + (scale) * acc[(2*i+1)*2*m + 2*j+1]));
		}
	}
}


void round_saturation_AVX(int16_t* acc,uint4x4_t* result_int_mat,float l_scale, float r_scale, float result_scale,uint4x4_t result_offset, int n, int m){

	float scale = l_scale * r_scale/result_scale;
	float zero_point = result_offset.i1;

	int j;
	__m256 scale_avx = _mm256_broadcast_ss(&scale);
	__m256 zp_avx = _mm256_broadcast_ss(&zero_point);
	__m256 upper_row;
	__m256 lower_row;

	uint32_t temp_upper[8];
	uint16_t fi[16];
	uint32_t temp_lower[8];
	uint16_t gi[16];
	float t1, t2, t3, t4;
	uint16_t temp;
	for(int i = 0; i<n; i++){
		for(j = 0; j<(m-7); j = j+8){
			upper_row = _mm256_loadu_ps((float *)(acc+(2*i*2*m+2*j)));
			lower_row = _mm256_loadu_ps((float *)(acc+((2*i+1)*2*m+2*j)));
			upper_row = _mm256_fmadd_ps(scale_avx,upper_row,zp_avx);
			lower_row = _mm256_fmadd_ps(scale_avx,lower_row,zp_avx);
			round_saturate_AVX(upper_row, lower_row, (__m256i*)&fi);
			
			temp = fi[0];
			result_int_mat[i*m + j].i1 = temp>>12;
			result_int_mat[i*m + j].i2 = temp>>8;
			result_int_mat[i*m + j].i3 = temp>>4;
			result_int_mat[i*m + j].i4 = temp;
			
			temp = fi[4];
			result_int_mat[i*m + j + 1].i1 = temp>>12;
			result_int_mat[i*m + j + 1].i2 = temp>>8;
			result_int_mat[i*m + j + 1].i3 = temp>>4;
			result_int_mat[i*m + j + 1].i4 = temp;
			
			temp = fi[8];
			result_int_mat[i*m + j + 2].i1 = temp>>12;
			result_int_mat[i*m + j + 2].i2 = temp>>8;
			result_int_mat[i*m + j + 2].i3 = temp>>4;
			result_int_mat[i*m + j + 2].i4 = temp;
			
			temp = fi[12];
			result_int_mat[i*m + j + 3].i1 = temp>>12;
			result_int_mat[i*m + j + 3].i2 = temp>>8;
			result_int_mat[i*m + j + 3].i3 = temp>>4;
			result_int_mat[i*m + j + 3].i4 = temp;
		}	
		for(; j<m; j = j+2){
			t1 = zero_point + acc[2*i*2*m + 2*j]*scale;
			t2 = zero_point + acc[2*i*2*m + 2*j + 1]*scale;
			t3 = zero_point + acc[2*(i + 1)*2*m + 2*j]*scale;
			t4 = zero_point + acc[2*(i + 1)*2*m + 2*j + 1]*scale;
			result_int_mat[i*m + j].i1 = (uint8_t)saturate(round(t1));
			result_int_mat[i*m + j].i2 = (uint8_t)saturate(round(t2));
			result_int_mat[i*m + j].i3 = (uint8_t)saturate(round(t3));
			result_int_mat[i*m + j].i4 = (uint8_t)saturate(round(t4));
		}
	}	

}