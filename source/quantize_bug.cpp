#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <limits.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"
#include "quantize.h"
#include <immintrin.h>

void vanilla_quantize(float *d, uint4x1_t *q, float *min, float *max, int rows, int columns){
	
	float scale, zero_point;
	get_min_max(d,rows,columns,min,max);
	quantize_parameter(*min,*max,&scale,&zero_point);

	float t;

	for(int i = 0; i<rows; i++){
		for(int j = 0; j<columns; j++){
			t = zero_point + d[i*columns + j]/scale;
			q[i*columns + j].i = (uint8_t)round(saturate(t));
		}
	}
}

void quantize_4x4(float *d, uint4x4_t *q, float *min, float *max, int rows, int columns){

	float scale, zero_point;
	get_min_max(d,rows,columns,min,max);
	quantize_parameter(*min,*max,&scale,&zero_point);

	float t1, t2, t3, t4;
	for(int i = 0; i<rows; i = i+2){
		for(int j = 0; j<columns; j = j+2){
			t1 = zero_point + d[i*columns + j]/scale;
			t2 = zero_point + d[i*columns + j + 1]/scale;
			t3 = zero_point + d[(i + 1)*columns + j]/scale;
			t4 = zero_point + d[(i + 1)*columns + j + 1]/scale;
			q[i/2*columns/2 + j/2].i1 = (uint8_t)saturate(round(t1));
			q[i/2*columns/2 + j/2].i2 = (uint8_t)saturate(round(t2));
			q[i/2*columns/2 + j/2].i3 = (uint8_t)saturate(round(t3));
			q[i/2*columns/2 + j/2].i4 = (uint8_t)saturate(round(t4));
		}
	}	
}

void quantize_AVX(float *d, uint4x4_t *q, float *min, float *max, int rows, int columns){

	float scale, zero_point;
	get_min_max_AVX(d,rows,columns,min,max);
	quantize_parameter(*min,*max,&scale,&zero_point);
	scale = 1/scale;
	float qmin = 0.0;
	float qmax = 15.0;
	int j;
	__m256 scale_avx = _mm256_broadcast_ss(&scale);
	__m256 zp_avx = _mm256_broadcast_ss(&zero_point);
	__m256 qmin_avx = _mm256_broadcast_ss(&qmin);
	__m256 qmax_avx = _mm256_broadcast_ss(&qmax);
	__m256 upper_row;
	__m256 lower_row;
	__m256i upper_rowi;
	__m256i lower_rowi;
	__m256 top_shift = _mm256_set_ps(256.0,4096.0,256.0,4096.0,256.0,4096.0,256.0,4096.0);
	__m256 bottom_shift = _mm256_set_ps(1.0,16.0,1.0,16.0,1.0,16.0,1.0,16.0);
	//__m256 top_shift = _mm256_set_epi32(1<<12,1<<8,2<<12,1<<8,1<<12,1<<8,1<<12,1<<8);
	//__m256 bottom_shift = _mm256_set_epi32(1<<4,1,1<<4,1,1<<4,1,1<<4,1);
	uint16_t temp_upper[16];
	//uint16_t temp_lower[16];
	float t1, t2, t3, t4;
	for(int i = 0; i<rows; i = i+2){
		for(j = 0; j<(columns-7); j = j+8){
			upper_row = _mm256_loadu_ps(d+(i*columns+j));
			lower_row = _mm256_loadu_ps(d+((i+1)*columns+j));

			upper_row = _mm256_fmadd_ps(scale_avx,upper_row,zp_avx);
			lower_row = _mm256_fmadd_ps(scale_avx,lower_row,zp_avx);

			//upper_row = _mm256_loadu_ps(f);
        		//lower_row = _mm256_loadu_ps(g);
        		//upper_row = _mm256_fmadd_ps(scale_avx,upper_row,zp_avx);
        		//lower_row = _mm256_fmadd_ps(scale_avx,lower_row,zp_avx);
        		upper_row = _mm256_max_ps( qmin_avx, _mm256_min_ps( qmax_avx, _mm256_round_ps(upper_row,_MM_FROUND_TO_NEAREST_INT)));
        		upper_row = _mm256_cvttps_epi32( _mm256_mul_ps( top_shift, upper_row) );
        		lower_row = _mm256_max_ps( qmin_avx, _mm256_min_ps( qmax_avx, _mm256_round_ps(lower_row,_MM_FROUND_TO_NEAREST_INT)));
        		lower_row = _mm256_cvttps_epi32( _mm256_mul_ps( bottom_shift, lower_row) );

        		//lower_row = _mm256_cvttps_epi32( bottom_shift, _mm256_cvttps_epi32( _mm256_max_ps( qmin_avx, _mm256_min_ps( qmax_avx, _mm256_round_ps(lower_row,_MM_FROUND_TO_NEAREST_INT)))));
        		upper_row = _mm256_or_si256( upper_row,lower_row);
        
        		//upper_rowi = _mm256_or_si256(upper_rowi, _mm256_slli_si256( upper_rowi,32));
        		_mm256_store_si256( (__m256i*)&temp_upper, upper_row);//_mm256_or_si256(upper_rowi,upper_rowi));
        		//_mm256_store_si256( (__m256i*)&gi, lower_row);//_mm256_or_si256(upper_rowi,upper_rowi));

			/*
			upper_row = _mm256_mul_epi32( top_shift, _mm256_cvttps_epi32( _mm256_max_ps( qmin_avx, _mm256_min_ps( qmax_avx, _mm256_round_ps(upper_row,_MM_FROUND_TO_NEAREST_INT)))));
			lower_row = _mm256_mul_epi32( bottom_shift, _mm256_cvttps_epi32( _mm256_max_ps( qmin_avx, _mm256_min_ps( qmax_avx, _mm256_round_ps(lower_row,_MM_FROUND_TO_NEAREST_INT)))));
			upper_rowi = _mm256_or_si256( _mm256_castps_si256(upper_row), _mm256_castps_si256(lower_row));
			upper_rowi = _mm256_or_si256(upper_rowi, _mm256_slli_si256( upper_rowi,32));
			*/
			/*__m256 t0 = _mm256_permute_ps(upper_row, 0x39);
			__m256 t1 = _mm256_permute2f128_ps(t0, t0, 0x81);
			upper_row = _mm256_blend_ps(t0, t1, 0x88);
			*/
			//upper_row = _mm256_or_si256(upper_row,upper_row);
			//_mm256_store_si256( (__m256i*)&temp_upper, _mm256_or_si256(upper_row,upper_row));
		
			*((int16_t*)&(*q)+(i/2*columns/2+j/2)) = (uint16_t) (temp_upper[0] | temp_upper[2]);
			*((uint16_t*)&(*q)+(i/2*columns/2+j/2+1)) = (uint16_t) (temp_upper[4] | temp_upper[6]);
			*((uint16_t*)&(*q)+(i/2*columns/2+j/2+2)) = (uint16_t) (temp_upper[8] | temp_upper[10]);
			*((uint16_t*)&(*q)+(i/2*columns/2+j/2+3)) = (uint16_t) (temp_upper[12] | temp_upper[14]);

		}
		for(; j<columns; j = j+2){
			t1 = zero_point + d[i*columns + j]*scale;
			t2 = zero_point + d[i*columns + j + 1]*scale;
			t3 = zero_point + d[(i + 1)*columns + j]*scale;
			t4 = zero_point + d[(i + 1)*columns + j + 1]*scale;
			q[i/2*columns/2 + j/2].i1 = (uint8_t)saturate(round(t1));
			q[i/2*columns/2 + j/2].i2 = (uint8_t)saturate(round(t2));
			q[i/2*columns/2 + j/2].i3 = (uint8_t)saturate(round(t3));
			q[i/2*columns/2 + j/2].i4 = (uint8_t)saturate(round(t4));
		}
	}	

}


void dequantize(double **d, uint8_t **q, double *mn, double *mx, int n){
	double delta = (*mx - *mn)/256; // size of linear cell
	double runner = 0.0;
	for(int i = 0; i<n; i++){
		for(int j = 0; j<n; j++){
			d[i][j] = *mn + q[i][j]*delta;
		}
	}
}
