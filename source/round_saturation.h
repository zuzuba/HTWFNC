#ifndef ROUND_SATURATION_H
#define ROUND_SATURATION_H

void round_saturation_naive(int16_t* acc,uint4x4_t* result_int_mat,float l_scale, float r_scale, float result_scale,uint4x4_t result_offset, int n, int m);
void round_saturation_AVX(int16_t* acc,uint4x4_t* result_int_mat,float l_scale, float r_scale, float result_scale,uint4x4_t result_offset, int n, int m);


#endif