#ifndef NAIVE_QMM_H
#define NAIVE_QMM_H

void qmm_space_waste(float l_scale, float r_scale, float result_scale, uint4x1_t l_offset, uint4x1_t r_offset, 
		uint4x1_t result_offset, uint4x1_t* l_int_mat, uint4x1_t* r_int_mat, uint4x1_t* result_int_mat, 
		int n, int k, int m);

void qmm_naive(float l_scale, float r_scale, float result_scale, uint4x4_t l_offset, uint4x4_t r_offset, 
	uint4x4_t result_offset, uint4x4_t* l_int_mat, uint4x4_t* r_int_mat, uint4x4_t* result_int_mat, 
	int n, int k, int m);

#endif