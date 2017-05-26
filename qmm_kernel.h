#ifndef QMM_KERNEL_H
#define QMM_KERNEL_H

void qmm_kernel_naive(uint4x4_t* l_int_mat, uint4x4_t* r_int_mat,uint4x4_t l_offset, uint4x4_t r_offset,int16_t* acc,int n,int k,int m);
void qmm_kernel_trick(uint4x4_t* l_int_mat, uint4x4_t* r_int_mat,int16_t* acc,int n,int k,int m);
void qmm_kernel_trick_blocking(uint4x4_t* l_int_mat, uint4x4_t* r_int_mat,int16_t* acc,int n,int k,int m);
void qmm_kernel_trick_AVX(uint4x4_t* l_int_mat, uint4x4_t* r_int_mat,int16_t* acc,int n,int k,int m);

#endif