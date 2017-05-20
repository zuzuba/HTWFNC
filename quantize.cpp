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
	scale = 1/scale;
	__m256 scale_avx = _mm256_broadcast_ss(&scale);
	__m256 zp_avx = _mm256_broadcast_ss(&zero_point);
	__m256 upper_row;
	__m256 lower_row;
	float *temp_upper = new float[8];
	float *temp_lower = new float[8];
	float t1, t2, t3, t4;
	for(int i = 0; i<rows; i = i+2){
		for(int j = 0; j<columns; j = j+8){
			upper_row = _mm256_loadu_ps(d+(i*columns+j));
			lower_row = _mm256_loadu_ps(d+((i+1)*columns+j));
			upper_row = _mm256_fmadd_ps(scale_avx,upper_row,zp_avx);
			lower_row = _mm256_fmadd_ps(scale_avx,lower_row,zp_avx);
			_mm256_store_ps(temp_upper,upper_row);
			_mm256_store_ps(temp_lower,lower_row);
			/*
			t1 = zero_point + d[i*columns + j]/scale;
			t2 = zero_point + d[i*columns + j + 1]/scale;
			t3 = zero_point + d[(i + 1)*columns + j]/scale;
			t4 = zero_point + d[(i + 1)*columns + j + 1]/scale;
			*/
			q[i/2*columns/2 + j/2].i1 = (uint8_t)saturate(round(temp_upper[0]));
			q[i/2*columns/2 + j/2].i2 = (uint8_t)saturate(round(temp_upper[1]));
			q[i/2*columns/2 + j/2].i3 = (uint8_t)saturate(round(temp_lower[0]));
			q[i/2*columns/2 + j/2].i4 = (uint8_t)saturate(round(temp_lower[1]));

			q[i/2*columns/2 + j/2+1].i1 = (uint8_t)saturate(round(temp_upper[2]));
			q[i/2*columns/2 + j/2+1].i2 = (uint8_t)saturate(round(temp_upper[3]));
			q[i/2*columns/2 + j/2+1].i3 = (uint8_t)saturate(round(temp_lower[2]));
			q[i/2*columns/2 + j/2+1].i4 = (uint8_t)saturate(round(temp_lower[3]));

			q[i/2*columns/2 + j/2+2].i1 = (uint8_t)saturate(round(temp_upper[4]));
			q[i/2*columns/2 + j/2+2].i2 = (uint8_t)saturate(round(temp_upper[5]));
			q[i/2*columns/2 + j/2+2].i3 = (uint8_t)saturate(round(temp_lower[4]));
			q[i/2*columns/2 + j/2+2].i4 = (uint8_t)saturate(round(temp_lower[5]));

			q[i/2*columns/2 + j/2+3].i1 = (uint8_t)saturate(round(temp_upper[6]));
			q[i/2*columns/2 + j/2+3].i2 = (uint8_t)saturate(round(temp_upper[7]));
			q[i/2*columns/2 + j/2+3].i3 = (uint8_t)saturate(round(temp_lower[6]));
			q[i/2*columns/2 + j/2+3].i4 = (uint8_t)saturate(round(temp_upper[7]));

		}
	}	

}


void dequantize(double **d, uint8_t **q, double *mn, double *mx, int n){
	double delta = (*mx - *mn)/256; // size of linear cell
	double runner = 0.0;
	for(int i = 0; i<n; i++)
		for(int j = 0; j<n; j++)
			d[i][j] = *mn + q[i][j]*delta; // if q[i][j] == 0, then d[i][j] == mn
}
