#include <stdio.h>
#include <stdint.h>
#include "utils.h"
#include "round_saturation.h"
#include <math.h>
#include <stdlib.h>
#include <immintrin.h> 

void round_saturation_naive(int16_t* acc,uint4x4_t* result_int_mat,float l_scale, float r_scale, float result_scale,uint4x4_t result_offset, int n, int m){

	for(int i=0; i<n; i = i+1){
		for(int j=0; j<m; j = j+1){
			result_int_mat[i*m + j].i1 = saturate(round(result_offset.i1 + (l_scale * r_scale/result_scale) * acc[(2*i)*2*m + 2*j]));
			result_int_mat[i*m + j].i2 = saturate(round(result_offset.i1 + (l_scale * r_scale/result_scale) * acc[(2*i)*2*m + 2*j+1]));
			result_int_mat[i*m + j].i3 = saturate(round(result_offset.i1 + (l_scale * r_scale/result_scale) * acc[(2*i+1)*2*m + 2*j]));
			result_int_mat[i*m + j].i4 = saturate(round(result_offset.i1 + (l_scale * r_scale/result_scale) * acc[(2*i+1)*2*m + 2*j+1]));
		}
	}
}