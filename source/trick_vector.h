#ifndef TRICK_VECTOR_H
#define TRICK_VECTOR_H

void trick_vector_naive(uint4x4_t* l_int_mat, uint4x4_t* r_int_mat,uint4x4_t l_offset, uint4x4_t r_offset,int n, int k, int m,
	uint16_t* term2, uint16_t* term3, uint16_t* term4);

void trick_vector_AVX(uint4x4_t* l_int_mat, uint4x4_t* r_int_mat,uint4x4_t l_offset, uint4x4_t r_offset,int n, int k, int m,
	uint16_t* term2, uint16_t* term3, uint16_t* term4);

#endif