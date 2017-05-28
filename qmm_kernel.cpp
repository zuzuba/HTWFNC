#include <stdio.h>
#include <stdint.h>
#include "utils.h"
#include "qmm_kernel.h"
#include <math.h>
#include <stdlib.h>
#include <immintrin.h>

void inline uint4x4_to_mm256_row_shuffle_inline(uint4x4_t* a, __m256i *b1, __m256i *b2){

	__m256i tmp = _mm256_loadu_si256((__m256i const *)a);

	__m256i mask13 = _mm256_set1_epi8(15);
	__m256i odd = _mm256_and_si256(tmp, mask13);
	
	__m256i mask24 = _mm256_set1_epi8(240);
	__m256i even = _mm256_and_si256(tmp, mask24);
	

	__m256i blend_mask = _mm256_set1_epi16(32768);
	*b1 = _mm256_blendv_epi8 (odd, _mm256_slli_epi64 (even, 4), blend_mask);
	*b2 = _mm256_blendv_epi8 (_mm256_srli_epi64 (odd, 8), _mm256_srli_epi64 (even, 4), blend_mask);

}

void transpose_inline(__m256i a0,__m256i a1,__m256i a2,__m256i a3,__m256i a4,__m256i a5,__m256i a6,__m256i a7,__m256i a8,
__m256i a9,__m256i a10,__m256i a11,__m256i a12,__m256i a13,__m256i a14,__m256i a15,__m256i a16,__m256i a17,__m256i a18,__m256i a19,__m256i a20,
__m256i a21,__m256i a22,__m256i a23,__m256i a24,__m256i a25,__m256i a26,__m256i a27,__m256i a28,__m256i a29,__m256i a30,__m256i a31,
 __m256i * a_t0,__m256i * a_t1,__m256i * a_t2,__m256i * a_t3,__m256i * a_t4,__m256i * a_t5,__m256i * a_t6,__m256i * a_t7,__m256i * a_t8,
__m256i * a_t9,__m256i * a_t10,__m256i * a_t11,__m256i * a_t12,__m256i * a_t13,__m256i * a_t14,__m256i * a_t15,__m256i * a_t16,__m256i * a_t17,__m256i * a_t18,__m256i * a_t19,__m256i * a_t20,
__m256i * a_t21,__m256i * a_t22,__m256i * a_t23,__m256i * a_t24,__m256i * a_t25,__m256i * a_t26,__m256i * a_t27,__m256i * a_t28,__m256i * a_t29,__m256i * a_t30,__m256i * a_t31){
	
	__m256i b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b24,b25,b26,b27,b28,b29,b30,b31;
	__m256i c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c24,c25,c26,c27,c28,c29,c30,c31;
	__m256i d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18,d19,d20,d21,d22,d23,d24,d25,d26,d27,d28,d29,d30,d31;
	__m256i e0,e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12,e13,e14,e15,e16,e17,e18,e19,e20,e21,e22,e23,e24,e25,e26,e27,e28,e29,e30,e31;
	

	b0  = _mm256_unpacklo_epi8 (a0, a1);   b1  = _mm256_unpackhi_epi8 (a0, a1);
	b2  = _mm256_unpacklo_epi8 (a2, a3);   b3  = _mm256_unpackhi_epi8 (a2, a3);
	b4  = _mm256_unpacklo_epi8 (a4, a5);   b5  = _mm256_unpackhi_epi8 (a4, a5);
	b6  = _mm256_unpacklo_epi8 (a6, a7);   b7  = _mm256_unpackhi_epi8 (a6, a7);
	b8  = _mm256_unpacklo_epi8 (a8, a9);   b9  = _mm256_unpackhi_epi8 (a8, a9);
	b10 = _mm256_unpacklo_epi8 (a10, a11); b11 = _mm256_unpackhi_epi8 (a10, a11);
	b12 = _mm256_unpacklo_epi8 (a12, a13); b13 = _mm256_unpackhi_epi8 (a12, a13);
	b14 = _mm256_unpacklo_epi8 (a14, a15); b15 = _mm256_unpackhi_epi8 (a14, a15);
	b16 = _mm256_unpacklo_epi8 (a16, a17); b17 = _mm256_unpackhi_epi8 (a16, a17);
	b18 = _mm256_unpacklo_epi8 (a18, a19); b19 = _mm256_unpackhi_epi8 (a18, a19);
	b20 = _mm256_unpacklo_epi8 (a20, a21); b21 = _mm256_unpackhi_epi8 (a20, a21);
	b22 = _mm256_unpacklo_epi8 (a22, a23); b23 = _mm256_unpackhi_epi8 (a22, a23);
	b24 = _mm256_unpacklo_epi8 (a24, a25); b25 = _mm256_unpackhi_epi8 (a24, a25);
	b26 = _mm256_unpacklo_epi8 (a26, a27); b27 = _mm256_unpackhi_epi8 (a26, a27);
	b28 = _mm256_unpacklo_epi8 (a28, a29); b29 = _mm256_unpackhi_epi8 (a28, a29);
	b30 = _mm256_unpacklo_epi8 (a30, a31); b31 = _mm256_unpackhi_epi8 (a30, a31);
	
	c0  = _mm256_unpacklo_epi16 (b0, b2);   c1  = _mm256_unpackhi_epi16 (b0, b2);
	c2  = _mm256_unpacklo_epi16 (b1, b3);   c3  = _mm256_unpackhi_epi16 (b1, b3);
	c4  = _mm256_unpacklo_epi16 (b4, b6);   c5  = _mm256_unpackhi_epi16 (b4, b6);
	c6  = _mm256_unpacklo_epi16 (b5, b7);   c7  = _mm256_unpackhi_epi16 (b5, b7);
	c8  = _mm256_unpacklo_epi16 (b8, b10);  c9  = _mm256_unpackhi_epi16 (b8, b10);
	c10 = _mm256_unpacklo_epi16 (b9, b11);  c11 = _mm256_unpackhi_epi16 (b9, b11);
	c12 = _mm256_unpacklo_epi16 (b12, b14); c13 = _mm256_unpackhi_epi16 (b12, b14);
	c14 = _mm256_unpacklo_epi16 (b13, b15); c15 = _mm256_unpackhi_epi16 (b13, b15);
	c16 = _mm256_unpacklo_epi16 (b16, b18); c17 = _mm256_unpackhi_epi16 (b16, b18);
	c18 = _mm256_unpacklo_epi16 (b17, b19); c19 = _mm256_unpackhi_epi16 (b17, b19);
	c20 = _mm256_unpacklo_epi16 (b20, b22); c21 = _mm256_unpackhi_epi16 (b20, b22);
	c22 = _mm256_unpacklo_epi16 (b21, b23); c23 = _mm256_unpackhi_epi16 (b21, b23);
	c24 = _mm256_unpacklo_epi16 (b24, b26); c25 = _mm256_unpackhi_epi16 (b24, b26);
	c26 = _mm256_unpacklo_epi16 (b25, b27); c27 = _mm256_unpackhi_epi16 (b25, b27);
	c28 = _mm256_unpacklo_epi16 (b28, b30); c29 = _mm256_unpackhi_epi16 (b28, b30);
	c30 = _mm256_unpacklo_epi16 (b29, b31); c31 = _mm256_unpackhi_epi16 (b29, b31);

	d0  = _mm256_unpacklo_epi32 (c0, c4);
	d1  = _mm256_unpackhi_epi32 (c0, c4);
	d2  = _mm256_unpacklo_epi32 (c1, c5);
	d3  = _mm256_unpackhi_epi32 (c1, c5);
	d4  = _mm256_unpacklo_epi32 (c2, c6);
	d5  = _mm256_unpackhi_epi32 (c2, c6);
	d6  = _mm256_unpacklo_epi32 (c3, c7);
	d7  = _mm256_unpackhi_epi32 (c3, c7);
	d8  = _mm256_unpacklo_epi32 (c8, c12);
	d9  = _mm256_unpackhi_epi32 (c8, c12);
	d10 = _mm256_unpacklo_epi32 (c9, c13);
	d11 = _mm256_unpackhi_epi32 (c9, c13);
	d12 = _mm256_unpacklo_epi32 (c10, c14);
	d13 = _mm256_unpackhi_epi32 (c10, c14);
	d14 = _mm256_unpacklo_epi32 (c11, c15);
	d15 = _mm256_unpackhi_epi32 (c11, c15);
	d16 = _mm256_unpacklo_epi32 (c16, c20);
	d17 = _mm256_unpackhi_epi32 (c16, c20);
	d18 = _mm256_unpacklo_epi32 (c17, c21);
	d19 = _mm256_unpackhi_epi32 (c17, c21);
	d20 = _mm256_unpacklo_epi32 (c18, c22);
	d21 = _mm256_unpackhi_epi32 (c18, c22);
	d22 = _mm256_unpacklo_epi32 (c19, c23);
	d23 = _mm256_unpackhi_epi32 (c19, c23);
	d24 = _mm256_unpacklo_epi32 (c24, c28);
	d25 = _mm256_unpackhi_epi32 (c24, c28);
	d26 = _mm256_unpacklo_epi32 (c25, c29);
	d27 = _mm256_unpackhi_epi32 (c25, c29);
	d28 = _mm256_unpacklo_epi32 (c26, c30);
	d29 = _mm256_unpackhi_epi32 (c26, c30);
	d30 = _mm256_unpacklo_epi32 (c27, c31);
	d31 = _mm256_unpackhi_epi32 (c27, c31);

	e0  = _mm256_unpacklo_epi64 (d0, d8);
	e1  = _mm256_unpackhi_epi64 (d0, d8);
	e2  = _mm256_unpacklo_epi64 (d1, d9);
	e3  = _mm256_unpackhi_epi64 (d1, d9);
	e4  = _mm256_unpacklo_epi64 (d2, d10);
	e5  = _mm256_unpackhi_epi64 (d2, d10);
	e6  = _mm256_unpacklo_epi64 (d3, d11);
	e7  = _mm256_unpackhi_epi64 (d3, d11);
	e8  = _mm256_unpacklo_epi64 (d4, d12);
	e9  = _mm256_unpackhi_epi64 (d4, d12);
	e10 = _mm256_unpacklo_epi64 (d5, d13);
	e11 = _mm256_unpackhi_epi64 (d5, d13);
	e12 = _mm256_unpacklo_epi64 (d6, d14);
	e13 = _mm256_unpackhi_epi64 (d6, d14);
	e14 = _mm256_unpacklo_epi64 (d7, d15);
	e15 = _mm256_unpackhi_epi64 (d7, d15);	
	e16 = _mm256_unpacklo_epi64 (d16, d24);
	e17 = _mm256_unpackhi_epi64 (d16, d24);
	e18 = _mm256_unpacklo_epi64 (d17, d25);
	e19 = _mm256_unpackhi_epi64 (d17, d25);
	e20 = _mm256_unpacklo_epi64 (d18, d26);
	e21 = _mm256_unpackhi_epi64 (d18, d26);
	e22 = _mm256_unpacklo_epi64 (d19, d27);
	e23 = _mm256_unpackhi_epi64 (d19, d27);
	e24 = _mm256_unpacklo_epi64 (d20, d28);
	e25 = _mm256_unpackhi_epi64 (d20, d28);
	e26 = _mm256_unpacklo_epi64 (d21, d29);
	e27 = _mm256_unpackhi_epi64 (d21, d29);
	e28 = _mm256_unpacklo_epi64 (d22, d30);
	e29 = _mm256_unpackhi_epi64 (d22, d30);
	e30 = _mm256_unpacklo_epi64 (d23, d31);
	e31 = _mm256_unpackhi_epi64 (d23, d31);

	*a_t0 =  _mm256_permute2f128_si256 (e0, e16, 32);
	*a_t16 =  _mm256_permute2f128_si256 (e0, e16, 50);
	*a_t1 =  _mm256_permute2f128_si256 (e1, e17, 32);
	*a_t17 =  _mm256_permute2f128_si256 (e1, e17, 50);
	*a_t2 =  _mm256_permute2f128_si256 (e2, e19, 32);
	*a_t18 =  _mm256_permute2f128_si256 (e2, e19, 50);
	*a_t3 =  _mm256_permute2f128_si256 (e3, e19, 32);
	*a_t19 =  _mm256_permute2f128_si256 (e3, e19, 50);
	*a_t4 =  _mm256_permute2f128_si256 (e4, e20, 32);
	*a_t20 =  _mm256_permute2f128_si256 (e4, e20, 50);
	*a_t5 =  _mm256_permute2f128_si256 (e5, e21, 32);
	*a_t21 =  _mm256_permute2f128_si256 (e5, e21, 50);
	*a_t6 =  _mm256_permute2f128_si256 (e6, e22, 32);
	*a_t22 =  _mm256_permute2f128_si256 (e6, e22, 50);
	*a_t7 =  _mm256_permute2f128_si256 (e7, e23, 32);
	*a_t23 =  _mm256_permute2f128_si256 (e7, e23, 50);	
	*a_t8 =  _mm256_permute2f128_si256 (e8, e24, 32);
	*a_t24 =  _mm256_permute2f128_si256 (e8, e24, 50);
	*a_t9 =  _mm256_permute2f128_si256 (e9, e25, 32);
	*a_t25 =  _mm256_permute2f128_si256 (e9, e25, 50);
	*a_t10 =  _mm256_permute2f128_si256 (e10, e26, 32);
	*a_t26 =  _mm256_permute2f128_si256 (e10, e26, 50);
	*a_t11 =  _mm256_permute2f128_si256 (e11, e27, 32);
	*a_t27 =  _mm256_permute2f128_si256 (e11, e27, 50);
	*a_t12 =  _mm256_permute2f128_si256 (e12, e28, 32);
	*a_t28 =  _mm256_permute2f128_si256 (e12, e28, 50);
	*a_t13 =  _mm256_permute2f128_si256 (e13, e29, 32);
	*a_t29 =  _mm256_permute2f128_si256 (e13, e29, 50);
	*a_t14 =  _mm256_permute2f128_si256 (e14, e30, 32);
	*a_t30 =  _mm256_permute2f128_si256 (e14, e30, 50);
	*a_t15 =  _mm256_permute2f128_si256 (e15, e31, 32);
	*a_t31 =  _mm256_permute2f128_si256 (e15, e31, 50);

}

void qmm_kernel_naive(uint4x4_t* l_int_mat, uint4x4_t* r_int_mat,uint4x4_t l_offset, uint4x4_t r_offset,int16_t* acc,int n,int k,int m){

	for(int i=0; i<n; i = i+1){
		for(int j=0; j<m; j = j+1){
			for (int t=0; t<k; t = t+1){
				acc[(2*i)*2*m + 2*j] += (l_int_mat[i*k + t].i1 - l_offset.i1) * (r_int_mat[t*m + j].i1 - r_offset.i1) + 
						(l_int_mat[i*k + t].i2 - l_offset.i1) * (r_int_mat[t*m + j].i3 - r_offset.i1);

				acc[(2*i)*2*m + 2*j+1] += (l_int_mat[i*k + t].i1 - l_offset.i1) * (r_int_mat[t*m + j].i2 - r_offset.i1) + 
						(l_int_mat[i*k + t].i2 - l_offset.i1) * (r_int_mat[t*m + j].i4 - r_offset.i1);

				acc[(2*i+1)*2*m + 2*j] += (l_int_mat[i*k + t].i3 - l_offset.i1) * (r_int_mat[t*m + j].i1 - r_offset.i1) + 
						(l_int_mat[i*k + t].i4 - l_offset.i1) * (r_int_mat[t*m + j].i3 - r_offset.i1);

				acc[(2*i+1)*2*m + 2*j +1] += (l_int_mat[i*k + t].i3 - l_offset.i1) * (r_int_mat[t*m + j].i2 - r_offset.i1) + 
						(l_int_mat[i*k + t].i4 - l_offset.i1) * (r_int_mat[t*m + j].i4 - r_offset.i1);
			}
		}
	}

}

void qmm_kernel_trick(uint4x4_t* l_int_mat, uint4x4_t* r_int_mat,int16_t* acc,int n,int k,int m){

	for(int i=0; i<n; i = i+1){
		for(int j=0; j<m; j = j+1){
			for (int t=0; t<k; t = t+1){
				acc[2*i*2*m + 2*j] += l_int_mat[i*k + t].i1*r_int_mat[t*m + j].i1 + l_int_mat[i*k + t].i2*r_int_mat[t*m + j].i3;

				acc[(2*i)*2*m + 2*j+1] += l_int_mat[i*k + t].i1*r_int_mat[t*m + j].i2 +l_int_mat[i*k + t].i2*r_int_mat[t*m + j].i4;

				acc[(2*i+1)*2*m + 2*j] += l_int_mat[i*k + t].i3*r_int_mat[t*m + j].i1 + l_int_mat[i*k + t].i4*r_int_mat[t*m + j].i3;

				acc[(2*i+1)*2*m + 2*j +1] += l_int_mat[i*k + t].i3*r_int_mat[t*m + j].i2 + l_int_mat[i*k + t].i4*r_int_mat[t*m + j].i4;
			}
		}
	}
}


void qmm_kernel_trick_blocking(uint4x4_t* l_int_mat, uint4x4_t* r_int_mat,int16_t* acc,int n,int k,int m){

	uint16_t term4,t001,t002,t003,t004,t101,t102,t103,t104,t011,t012,t013,t014,t021,t022,t023,t024,t111,t112,t113,t114,t121,t122,t123,t124,t201,t202,t203,t204;
	uint16_t t211,t212,t213,t214,t221,t222,t223,t224;
	uint4x4_t r_elment,l_element,l0,l1,l2,r0,r1,r2;
	int Nb=21;
	int i,j,t,i_f,j_f,t_f;
	for(i=0; i<n-Nb+1; i +=Nb){
		for(j=0; j<m-Nb+1; j += Nb){
			for (t=0; t<k-Nb+1; t +=Nb){
				for(int i_p = i;i_p<i+Nb;i_p += 3){
					for(int j_p=j;j_p<j+Nb;j_p += 3){
						//LR
						//loads of the accumulator matrix
						t001 = acc[2*i_p*2*m + 2*j_p];
						t002 = acc[2*i_p*2*m + 2*j_p+1];
						t003 = acc[(2*i_p+1)*2*m + 2*j_p];
						t004 = acc[(2*i_p+1)*2*m + 2*j_p+1];

						t011 = acc[2*i_p*2*m + 2*(j_p+1)];
						t012 = acc[2*i_p*2*m + 2*(j_p+1)+1];
						t013 = acc[(2*i_p+1)*2*m + 2*(j_p+1)];
						t014 = acc[(2*i_p+1)*2*m + 2*(j_p+1)+1];

						t021 = acc[2*i_p*2*m + 2*(j_p+2)];
						t022 = acc[2*i_p*2*m + 2*(j_p+2)+1];
						t023 = acc[(2*i_p+1)*2*m + 2*(j_p+2)];
						t024 = acc[(2*i_p+1)*2*m + 2*(j_p+2)+1];

						t101 = acc[(2*(i_p+1))*2*m + 2*j_p];
						t102 = acc[(2*(i_p+1))*2*m + 2*j_p+1];
						t103 = acc[(2*(i_p+1)+1)*2*m + 2*j_p];
						t104 = acc[(2*(i_p+1)+1)*2*m + 2*j_p+1];

						t111 = acc[(2*(i_p+1))*2*m + 2*(j_p+1)];
						t112 = acc[(2*(i_p+1))*2*m + 2*(j_p+1)+1];
						t113 = acc[(2*(i_p+1)+1)*2*m + 2*(j_p+1)];
						t114 = acc[(2*(i_p+1)+1)*2*m + 2*(j_p+1)+1];

						t121 = acc[(2*(i_p+1))*2*m + 2*(j_p+2)];
						t122 = acc[(2*(i_p+1))*2*m + 2*(j_p+2)+1];
						t123 = acc[(2*(i_p+1)+1)*2*m + 2*(j_p+2)];
						t124 = acc[(2*(i_p+1)+1)*2*m + 2*(j_p+2)+1];

						t201 = acc[(2*(i_p+2))*2*m + 2*(j_p)];
						t202 = acc[(2*(i_p+2))*2*m + 2*(j_p)+1];
						t203 = acc[(2*(i_p+2)+1)*2*m + 2*(j_p)];
						t204 = acc[(2*(i_p+2)+1)*2*m + 2*(j_p)+1];

						t211 = acc[(2*(i_p+2))*2*m + 2*(j_p+1)];
						t212 = acc[(2*(i_p+2))*2*m + 2*(j_p+1)+1];
						t213 = acc[(2*(i_p+2)+1)*2*m + 2*(j_p+1)];
						t214 = acc[(2*(i_p+2)+1)*2*m + 2*(j_p+1)+1];

						t221 = acc[(2*(i_p+2))*2*m + 2*(j_p+2)];
						t222 = acc[(2*(i_p+2))*2*m + 2*(j_p+2)+1];
						t223 = acc[(2*(i_p+2)+1)*2*m + 2*(j_p+2)];
						t224 = acc[(2*(i_p+2)+1)*2*m + 2*(j_p+2)+1];

						for(int t_p=t;t_p<t+Nb;t_p += 3){
							//loads of left and right hand side
							l0 = l_int_mat[i_p*k + t_p];
							l1 = l_int_mat[(i_p+1)*k + t_p];
							l2 = l_int_mat[(i_p+2)*k + t_p];
							r0 = r_int_mat[t_p*m + j_p];
							r1 = r_int_mat[t_p*m + j_p+1];
							r2 = r_int_mat[t_p*m + j_p+2];

							//computation
							t001 += l0.i1*r0.i1 + l0.i2*r0.i3;
							t002 += l0.i1*r0.i2 + l0.i2*r0.i4;
							t003 += l0.i3*r0.i1 + l0.i4*r0.i3;
							t004 += l0.i3*r0.i2 + l0.i4*r0.i4;
							
							t011 += l0.i1*r1.i1 + l0.i2*r1.i3;
							t012 += l0.i1*r1.i2 + l0.i2*r1.i4;
							t013 += l0.i3*r1.i1 + l0.i4*r1.i3;
							t014 += l0.i3*r1.i2 + l0.i4*r1.i4;

							t021 += l0.i1*r2.i1 + l0.i2*r2.i3;
							t022 += l0.i1*r2.i2 + l0.i2*r2.i4;
							t023 += l0.i3*r2.i1 + l0.i4*r2.i3;
							t024 += l0.i3*r2.i2 + l0.i4*r2.i4;

							t101 += l1.i1*r0.i1 + l1.i2*r0.i3;
							t102 += l1.i1*r0.i2 + l1.i2*r0.i4;
							t103 += l1.i3*r0.i1 + l1.i4*r0.i3;
							t104 += l1.i3*r0.i2 + l1.i4*r0.i4;

							t111 += l1.i1*r1.i1 + l1.i2*r1.i3;
							t112 += l1.i1*r1.i2 + l1.i2*r1.i4;
							t113 += l1.i3*r1.i1 + l1.i4*r1.i3;
							t114 += l1.i3*r1.i2 + l1.i4*r1.i4;

							t121 += l1.i1*r2.i1 + l1.i2*r2.i3;
							t122 += l1.i1*r2.i2 + l1.i2*r2.i4;
							t123 += l1.i3*r2.i1 + l1.i4*r2.i3;
							t124 += l1.i3*r2.i2 + l1.i4*r2.i4;

							t201 += l2.i1*r0.i1 + l2.i2*r0.i3;
							t202 += l2.i1*r0.i2 + l2.i2*r0.i4;
							t203 += l2.i3*r0.i1 + l2.i4*r0.i3;
							t204 += l2.i3*r0.i2 + l2.i4*r0.i4;

							t211 += l2.i1*r1.i1 + l2.i2*r1.i3;
							t212 += l2.i1*r1.i2 + l2.i2*r1.i4;
							t213 += l2.i3*r1.i1 + l2.i4*r1.i3;
							t214 += l2.i3*r1.i2 + l2.i4*r1.i4;

							t221 += l2.i1*r2.i1 + l2.i2*r2.i3;
							t222 += l2.i1*r2.i2 + l2.i2*r2.i4;
							t223 += l2.i3*r2.i1 + l2.i4*r2.i3;
							t224 += l2.i3*r2.i2 + l2.i4*r2.i4;
						}

						//store
						acc[2*i_p*2*m + 2*j_p] = t001;
						acc[2*i_p*2*m + 2*j_p+1] = t002;
						acc[(2*i_p+1)*2*m + 2*j_p] = t003;
						acc[(2*i_p+1)*2*m + 2*j_p+1] = t004;

						acc[2*i_p*2*m + 2*(j_p+1)] = t011;
						acc[2*i_p*2*m + 2*(j_p+1)+1] = t012;
						acc[(2*i_p+1)*2*m + 2*(j_p+1)] = t013;
						acc[(2*i_p+1)*2*m + 2*(j_p+1)+1] = t014;

						acc[2*i_p*2*m + 2*(j_p+2)]  = t021;
						acc[2*i_p*2*m + 2*(j_p+2)+1] = t022;
						acc[(2*i_p+1)*2*m + 2*(j_p+2)] = t023;
						acc[(2*i_p+1)*2*m + 2*(j_p+2)+1] = t024;

						acc[(2*(i_p+1))*2*m + 2*j_p] = t101;
						acc[(2*(i_p+1))*2*m + 2*j_p+1] = t102;
					    acc[(2*(i_p+1)+1)*2*m + 2*j_p] = t103;
						acc[(2*(i_p+1)+1)*2*m + 2*j_p+1] = t104;

						acc[(2*(i_p+1))*2*m + 2*(j_p+1)] = t111;
						acc[(2*(i_p+1))*2*m + 2*(j_p+1)+1] = t112;
						acc[(2*(i_p+1)+1)*2*m + 2*(j_p+1)] = t113;
						acc[(2*(i_p+1)+1)*2*m + 2*(j_p+1)+1] = t114;

						acc[(2*(i_p+1))*2*m + 2*(j_p+2)] = t121;
						acc[(2*(i_p+1))*2*m + 2*(j_p+2)+1] = t122;
						acc[(2*(i_p+1)+1)*2*m + 2*(j_p+2)] = t123;
						acc[(2*(i_p+1)+1)*2*m + 2*(j_p+2)+1] = t124;

						acc[(2*(i_p+2))*2*m + 2*(j_p)] = t201;
						acc[(2*(i_p+2))*2*m + 2*(j_p)+1] = t202;
						acc[(2*(i_p+2)+1)*2*m + 2*(j_p)] = t203;
						acc[(2*(i_p+2)+1)*2*m + 2*(j_p)+1] = t204;

						acc[(2*(i_p+2))*2*m + 2*(j_p+1)] = t211;
						acc[(2*(i_p+2))*2*m + 2*(j_p+1)+1] = t212;
						acc[(2*(i_p+2)+1)*2*m + 2*(j_p+1)] = t213;
						acc[(2*(i_p+2)+1)*2*m + 2*(j_p+1)+1] = t214;

						acc[(2*(i_p+2))*2*m + 2*(j_p+2)] = t221;
						acc[(2*(i_p+2))*2*m + 2*(j_p+2)+1] = t222;
						acc[(2*(i_p+2)+1)*2*m + 2*(j_p+2)] = t223;
						acc[(2*(i_p+2)+1)*2*m + 2*(j_p+2)+1] = t224;
					}
				}
				t_f = t;
			}
			j_f=j;
		}
		i_f=i;
	}

	//LcRr
	for (i = i_f; i < n; ++i){
		for ( j = j_f; j < m; ++j){
			for (; t<k; t++ ){
				acc[2*i*2*m + 2*j] += l_int_mat[i*k + t].i1*r_int_mat[t*m + j].i1 + l_int_mat[i*k + t].i2*r_int_mat[t*m + j].i3;
				acc[2*i*2*m + 2*j+1] += l_int_mat[i*k + t].i1*r_int_mat[t*m + j].i2 + l_int_mat[i*k + t].i2*r_int_mat[t*m + j].i4;
				acc[(2*i+1)*2*m + 2*j] += l_int_mat[i*k + t].i3*r_int_mat[t*m + j].i1 + l_int_mat[i*k + t].i4*r_int_mat[t*m + j].i3;
				acc[(2*i+1)*2*m + 2*j+1] += l_int_mat[i*k + t].i3*r_int_mat[t*m + j].i2 + l_int_mat[i*k + t].i4*r_int_mat[t*m + j].i4;
			}		
		}
	}

	//LRc + LcRs
	for(i=0;i<n-Nb+1;i++){
		for (j=j_f; j < m; j++){
			for (t=0; t<k; t++ ){
				acc[2*i*2*m + 2*j] += l_int_mat[i*k + t].i1*r_int_mat[t*m + j].i1 + l_int_mat[i*k + t].i2*r_int_mat[t*m + j].i3;
				acc[2*i*2*m + 2*j+1] += l_int_mat[i*k + t].i1*r_int_mat[t*m + j].i2 + l_int_mat[i*k + t].i2*r_int_mat[t*m + j].i4;
				acc[(2*i+1)*2*m + 2*j] += l_int_mat[i*k + t].i3*r_int_mat[t*m + j].i1 + l_int_mat[i*k + t].i4*r_int_mat[t*m + j].i3;
				acc[(2*i+1)*2*m + 2*j+1] += l_int_mat[i*k + t].i3*r_int_mat[t*m + j].i2 + l_int_mat[i*k + t].i4*r_int_mat[t*m + j].i4;
			}
		}
	}

	//LrR + LsRr  || LrRc + LsRs
	for (i=i_f; i < n; i++){
		for (j=0; j < m; j++){
			for (t=0; t<k; t++ ){
				acc[2*i*2*m + 2*j] += l_int_mat[i*k + t].i1*r_int_mat[t*m + j].i1 + l_int_mat[i*k + t].i2*r_int_mat[t*m + j].i3;
				acc[2*i*2*m + 2*j+1] += l_int_mat[i*k + t].i1*r_int_mat[t*m + j].i2 + l_int_mat[i*k + t].i2*r_int_mat[t*m + j].i4;
				acc[(2*i+1)*2*m + 2*j] += l_int_mat[i*k + t].i3*r_int_mat[t*m + j].i1 + l_int_mat[i*k + t].i4*r_int_mat[t*m + j].i3;
				acc[(2*i+1)*2*m + 2*j+1] += l_int_mat[i*k + t].i3*r_int_mat[t*m + j].i2 + l_int_mat[i*k + t].i4*r_int_mat[t*m + j].i4;
			}
		}
	}
}	

void qmm_kernel_trick_AVX(uint4x4_t* l_int_mat, uint4x4_t* r_int_mat,int16_t* acc,int n,int k,int m){
	
	__m256i r1,r2;
	__m256i temp[32],temp_t[32];
	__m256i acc11[16],acc12[16],acc21[16],acc22[16],dot_prod1[32],dot_prod2[32];
	uint16_t acc11_int[16],acc12_int[16],acc21_int[16],acc22_int[16];
	int i,j,t;
	uint16_t acc1,acc2,acc3,acc4;

	for( i=0; i<n; i = i+1){
		for( j=0; j<(m-15); j = j+16){	
			for (int u = 0; u < 16; u++){
				acc11_int[u] = 0;
				acc12_int[u] = 0;
				acc21_int[u] = 0;
				acc22_int[u] = 0;
				acc11[u] = _mm256_set1_epi16(0);
				acc12[u] = _mm256_set1_epi16(0);
				acc21[u] = _mm256_set1_epi16(0);
				acc22[u] = _mm256_set1_epi16(0);	
			}

			for(t=0; t<k-31; t+=32){
				uint4x4_to_mm256_row_shuffle(l_int_mat + i*k + t, &r1, &r2);
				
				for (int u = 0; u < 16; u++){
					uint4x4_to_mm256_row_shuffle(r_int_mat + (t+u)*m + j, &temp[2*u], &temp[2*u+1]);	
				}
				transpose(temp,temp_t);

				for (int u = 0; u < 32; u++){
					dot_prod1[u] = _mm256_maddubs_epi16 (r1,temp_t[u]);
					dot_prod2[u] = _mm256_maddubs_epi16 (r2,temp_t[u]);
				}

				for (int u = 0; u < 16; u++){
					acc11[u] = _mm256_add_epi16(acc11[u],dot_prod1[2*u]);
					acc12[u] = _mm256_add_epi16(acc12[u],dot_prod1[2*u+1]);
					acc21[u] = _mm256_add_epi16(acc21[u],dot_prod2[2*u]);
					acc22[u] = _mm256_add_epi16(acc22[u],dot_prod2[2*u+1]);
				}
				
			}

			for (int u = 0; u < 16; u++){
				acc11_int[u] += _mm256_haddsi_epi16(acc11[u]);
				acc12_int[u] += _mm256_haddsi_epi16(acc12[u]);
				acc21_int[u] += _mm256_haddsi_epi16(acc21[u]);
				acc22_int[u] += _mm256_haddsi_epi16(acc22[u]);
			}

			for (int t = 0; t < 16; t++){
				for(int u=t;u<k; u = u+1){
					acc11_int[t] += (l_int_mat[i*k + u].i1) * (r_int_mat[u*m + j+t].i1) + 
							(l_int_mat[i*k + u].i2) * (r_int_mat[u*m + j].i3);

					acc12_int[t] += (l_int_mat[i*k + u].i1) * (r_int_mat[u*m + j+t].i2) + 
							(l_int_mat[i*k + u].i2) * (r_int_mat[u*m + j+t].i4);

					acc21_int[t] += (l_int_mat[i*k + u].i3) * (r_int_mat[u*m + j+t].i1) + 
							(l_int_mat[i*k + u].i4) * (r_int_mat[u*m + j].i3);

					acc22_int[t] += (l_int_mat[i*k + u].i3) * (r_int_mat[u*m + j+t].i2) + 
							(l_int_mat[i*k + u].i4) * (r_int_mat[u*m + j+t].i4);

				}
			}

			for (int u = 0; u < 8; u+=1){
				acc[(2*i)*2*m + 2*j+2*u] = acc11_int[2*u];
				acc[(2*i)*2*m + 2*j+2*u+1] = acc12_int[2*u];
				acc[(2*i+1)*2*m + 2*j+2*u] = acc21_int[2*u];
				acc[(2*i+1)*2*m + 2*j+2*u+1] = acc22_int[2*u];
				acc[(2*i)*2*m + 2*j+2*u+2] = acc11_int[2*u+1];
				acc[(2*i)*2*m + 2*j+2*u+3] = acc12_int[2*u+1];
				acc[(2*i+1)*2*m + 2*j+2*u+2] = acc21_int[2*u+1];
				acc[(2*i+1)*2*m + 2*j+2*u+3] = acc22_int[2*u+1];
			}
		}

		for(; j<m; j = j+1){
			acc1 = 0;
			acc2 = 0;
			acc3 = 0;
			acc4 = 0;
			for (int t=0; t<k; t = t+1){
				uint4x4_t l_element = l_int_mat[i*k + t]; 
				uint4x4_t r_elment = r_int_mat[t*m + j];
				acc1 += (l_element.i1) * (r_elment.i1) + 
						(l_element.i2) * (r_elment.i3);

				acc2 += (l_element.i1) * (r_elment.i2) + 
						(l_element.i2) * (r_elment.i4);

				acc3 += (l_element.i3) * (r_elment.i1) + 
						(l_element.i4) * (r_elment.i3);

				acc4 += (l_element.i3) * (r_elment.i2) + 
						(l_element.i4) * (r_elment.i4);

			}

			acc[(2*i)*2*m + 2*j] = acc1;
			acc[(2*i)*2*m + 2*j+1] = acc2;
			acc[(2*i+1)*2*m + 2*j] = acc3;
			acc[(2*i+1)*2*m + 2*j+1] = acc4;
		}
	}
}



void qmm_kernel_trick_AVX_unrolled(uint4x4_t* l_int_mat, uint4x4_t* r_int_mat,int16_t* acc,int n,int k,int m){
	
	__m256i r1,r2;
	__m256i temp0,temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp8,temp9,temp10,temp11,temp12,temp13,temp14,temp15,temp16,temp17,temp18,temp19,temp20,temp21;
	__m256i temp22,temp23,temp24,temp25,temp26,temp27,temp28,temp29,temp30,temp31;
	__m256i temp_t0,temp_t1,temp_t2,temp_t3,temp_t4,temp_t5,temp_t6,temp_t7,temp_t8,temp_t9,temp_t10,temp_t11,temp_t12,temp_t13,temp_t14,temp_t15,temp_t16,temp_t17,temp_t18,temp_t19,temp_t20,temp_t21;
	__m256i temp_t22,temp_t23,temp_t24,temp_t25,temp_t26,temp_t27,temp_t28,temp_t29,temp_t30,temp_t31;
	__m256i acc11_0,acc11_1,acc11_2,acc11_3,acc11_4,acc11_5,acc11_6,acc11_7,acc11_8,acc11_9,acc11_10,acc11_11,acc11_12,acc11_13,acc11_14,acc11_15;
	__m256i acc12_0,acc12_1,acc12_2,acc12_3,acc12_4,acc12_5,acc12_6,acc12_7,acc12_8,acc12_9,acc12_10,acc12_11,acc12_12,acc12_13,acc12_14,acc12_15;
	__m256i acc21_0,acc21_1,acc21_2,acc21_3,acc21_4,acc21_5,acc21_6,acc21_7,acc21_8,acc21_9,acc21_10,acc21_11,acc21_12,acc21_13,acc21_14,acc21_15;
	__m256i acc22_0,acc22_1,acc22_2,acc22_3,acc22_4,acc22_5,acc22_6,acc22_7,acc22_8,acc22_9,acc22_10,acc22_11,acc22_12,acc22_13,acc22_14,acc22_15;
	__m256i dot_prod10,dot_prod11,dot_prod12,dot_prod13,dot_prod14,dot_prod15,dot_prod16,dot_prod17,dot_prod18,dot_prod19,dot_prod110,dot_prod111;
	__m256i dot_prod112,dot_prod113,dot_prod114,dot_prod115,dot_prod116,dot_prod117,dot_prod118,dot_prod119,dot_prod120,dot_prod121,dot_prod122,dot_prod123,dot_prod124;
	__m256i dot_prod125,dot_prod126,dot_prod127,dot_prod128,dot_prod129,dot_prod130,dot_prod131;
	__m256i dot_prod20,dot_prod21,dot_prod22,dot_prod23,dot_prod24,dot_prod25,dot_prod26,dot_prod27,dot_prod28,dot_prod29,dot_prod210,dot_prod211,dot_prod212,dot_prod213;
	__m256i dot_prod214,dot_prod215,dot_prod216,dot_prod217,dot_prod218,dot_prod219,dot_prod220,dot_prod221,dot_prod222,dot_prod223,dot_prod224,dot_prod225,dot_prod226,dot_prod227,dot_prod228;
	__m256i dot_prod229,dot_prod230,dot_prod231;
	uint16_t acc11_int0,acc11_int1,acc11_int2,acc11_int3,acc11_int4,acc11_int5,acc11_int6,acc11_int7,acc11_int8,acc11_int9,acc11_int10,acc11_int11,acc11_int12;
	uint16_t acc11_int13,acc11_int14,acc11_int15,acc12_int0,acc12_int1,acc12_int2,acc12_int3,v,acc12_int4,acc12_int5,acc12_int6,acc12_int7,acc12_int8;
	uint16_t acc12_int9,acc12_int10,acc12_int11,acc12_int12,acc12_int13,acc12_int14,acc12_int15,acc21_int0,acc21_int1,acc21_int2,acc21_int3,acc21_int4,acc21_int5,acc21_int6,acc21_int7;
	uint16_t acc21_int8,acc21_int9,acc21_int10,acc21_int11,acc21_int12,acc21_int13,acc21_int14,acc21_int15,acc22_int0,acc22_int1,acc22_int2,acc22_int3,acc22_int4;
	uint16_t acc22_int5,acc22_int6,acc22_int7,acc22_int8,acc22_int9,acc22_int10,acc22_int11,acc22_int12,acc22_int13,acc22_int14,acc22_int15;
	int i,j,t;
	uint16_t acc1,acc2,acc3,acc4;
	__m256i mask13 = _mm256_set1_epi8(15);
	__m256i mask24 = _mm256_set1_epi8(240);
	__m256i blend_mask = _mm256_set1_epi16(32768);
	__m256i tmp, odd, even;

	for( i=0; i<n; i = i+1){
		for( j=0; j<(m-15); j = j+16){	
			for (int u = 0; u < 16; u++){
				acc11_int0 = 0; acc11_int1 = 0; acc11_int2 = 0; acc11_int3 = 0;
				acc11_int4 = 0; acc11_int5 = 0; acc11_int6 = 0; acc11_int7 = 0;
				acc11_int8 = 0; acc11_int9 = 0; acc11_int10 = 0; acc11_int11 = 0;
				acc11_int12 = 0; acc11_int13 = 0; acc11_int14 = 0; acc11_int15 = 0;

				acc12_int0 = 0; acc12_int1 = 0; acc12_int2 = 0; acc12_int3 = 0;
				acc12_int4 = 0; acc12_int5 = 0; acc12_int6 = 0; acc12_int7 = 0;
				acc12_int8 = 0; acc12_int9 = 0; acc12_int10 = 0; acc12_int11 = 0;
				acc12_int12 = 0; acc12_int13 = 0; acc12_int14 = 0; acc12_int15 = 0;

				acc21_int0 = 0; acc21_int1 = 0; acc21_int2 = 0; acc21_int3 = 0;
				acc21_int4 = 0; acc21_int5 = 0; acc21_int6 = 0; acc21_int7 = 0;
				acc21_int8 = 0; acc21_int9 = 0; acc21_int10 = 0; acc21_int11 = 0;
				acc21_int12 = 0; acc21_int13 = 0; acc21_int14 = 0; acc21_int15 = 0;

				acc22_int0 = 0; acc22_int1 = 0; acc22_int2 = 0; acc22_int3 = 0;
				acc22_int4 = 0; acc22_int5 = 0; acc22_int6 = 0; acc22_int7 = 0;
				acc22_int8 = 0; acc22_int9 = 0; acc22_int10 = 0; acc22_int11 = 0;
				acc22_int12 = 0; acc22_int13 = 0; acc22_int14 = 0; acc22_int15 = 0;
				
				acc11_0  = _mm256_set1_epi16(0); acc11_1  = _mm256_set1_epi16(0); acc11_2  = _mm256_set1_epi16(0); acc11_3  = _mm256_set1_epi16(0);
				acc11_4  = _mm256_set1_epi16(0); acc11_5  = _mm256_set1_epi16(0); acc11_6  = _mm256_set1_epi16(0); acc11_7  = _mm256_set1_epi16(0);
				acc11_8  = _mm256_set1_epi16(0); acc11_9  = _mm256_set1_epi16(0); acc11_10 = _mm256_set1_epi16(0); acc11_11 = _mm256_set1_epi16(0);
				acc11_12 = _mm256_set1_epi16(0); acc11_13 = _mm256_set1_epi16(0); acc11_14 = _mm256_set1_epi16(0); acc11_15 = _mm256_set1_epi16(0);

				acc12_0  = _mm256_set1_epi16(0); acc12_1  = _mm256_set1_epi16(0); acc12_2  = _mm256_set1_epi16(0); acc12_3  = _mm256_set1_epi16(0);
				acc12_4  = _mm256_set1_epi16(0); acc12_5  = _mm256_set1_epi16(0); acc12_6  = _mm256_set1_epi16(0); acc12_7  = _mm256_set1_epi16(0);
				acc12_8  = _mm256_set1_epi16(0); acc12_9  = _mm256_set1_epi16(0); acc12_10 = _mm256_set1_epi16(0); acc12_11 = _mm256_set1_epi16(0);
				acc12_12 = _mm256_set1_epi16(0); acc12_13 = _mm256_set1_epi16(0); acc12_14 = _mm256_set1_epi16(0); acc12_15 = _mm256_set1_epi16(0);

				acc21_0  = _mm256_set1_epi16(0); acc21_1  = _mm256_set1_epi16(0); acc21_2  = _mm256_set1_epi16(0); acc21_3  = _mm256_set1_epi16(0);
				acc21_4  = _mm256_set1_epi16(0); acc21_5  = _mm256_set1_epi16(0); acc21_6  = _mm256_set1_epi16(0); acc21_7  = _mm256_set1_epi16(0);
				acc21_8  = _mm256_set1_epi16(0); acc21_9  = _mm256_set1_epi16(0); acc21_10 = _mm256_set1_epi16(0); acc21_11 = _mm256_set1_epi16(0);
				acc21_12 = _mm256_set1_epi16(0); acc21_13 = _mm256_set1_epi16(0); acc21_14 = _mm256_set1_epi16(0); acc21_15 = _mm256_set1_epi16(0);

				acc22_0  = _mm256_set1_epi16(0); acc22_1  = _mm256_set1_epi16(0); acc22_2  = _mm256_set1_epi16(0); acc22_3  = _mm256_set1_epi16(0);
				acc22_4  = _mm256_set1_epi16(0); acc22_5  = _mm256_set1_epi16(0); acc22_6  = _mm256_set1_epi16(0); acc22_7  = _mm256_set1_epi16(0);
				acc22_8  = _mm256_set1_epi16(0); acc22_9  = _mm256_set1_epi16(0); acc22_10 = _mm256_set1_epi16(0); acc22_11 = _mm256_set1_epi16(0);
				acc22_12 = _mm256_set1_epi16(0); acc22_13 = _mm256_set1_epi16(0); acc22_14 = _mm256_set1_epi16(0); acc22_15 = _mm256_set1_epi16(0);

			}

			for ( t=0; t<(k-31); t = t+32){
				uint4x4_to_mm256_row_shuffle_inline(l_int_mat + i*k + t, &r1, &r2);
				uint4x4_to_mm256_row_shuffle_inline(r_int_mat + (t+0)*m + j, &temp0, &temp1);
				uint4x4_to_mm256_row_shuffle_inline(r_int_mat + (t+1)*m + j, &temp2, &temp3);
				uint4x4_to_mm256_row_shuffle_inline(r_int_mat + (t+2)*m + j, &temp4, &temp5);
				uint4x4_to_mm256_row_shuffle_inline(r_int_mat + (t+3)*m + j, &temp6, &temp7);
				uint4x4_to_mm256_row_shuffle_inline(r_int_mat + (t+4)*m + j, &temp8, &temp9);
				uint4x4_to_mm256_row_shuffle_inline(r_int_mat + (t+5)*m + j, &temp10, &temp11);
				uint4x4_to_mm256_row_shuffle_inline(r_int_mat + (t+6)*m + j, &temp12, &temp13);
				uint4x4_to_mm256_row_shuffle_inline(r_int_mat + (t+7)*m + j, &temp14, &temp15);
				uint4x4_to_mm256_row_shuffle_inline(r_int_mat + (t+8)*m + j, &temp16, &temp17);
				uint4x4_to_mm256_row_shuffle_inline(r_int_mat + (t+9)*m + j, &temp18, &temp19);
				uint4x4_to_mm256_row_shuffle_inline(r_int_mat + (t+10)*m + j, &temp20, &temp21);
				uint4x4_to_mm256_row_shuffle_inline(r_int_mat + (t+11)*m + j, &temp22, &temp23);
				uint4x4_to_mm256_row_shuffle_inline(r_int_mat + (t+12)*m + j, &temp24, &temp25);
				uint4x4_to_mm256_row_shuffle_inline(r_int_mat + (t+13)*m + j, &temp26, &temp27);
				uint4x4_to_mm256_row_shuffle_inline(r_int_mat + (t+14)*m + j, &temp28, &temp29);
				uint4x4_to_mm256_row_shuffle_inline(r_int_mat + (t+15)*m + j, &temp30, &temp31);
				
				transpose_inline( temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8,temp9, temp10, temp11, temp12, temp13, temp14, temp15, temp16, temp17, temp18, temp19, temp20,
					temp21, temp22, temp23, temp24, temp25, temp26, temp27, temp28, temp29, temp30, temp31, &temp_t0, &temp_t1, &temp_t2, &temp_t3, &temp_t4, &temp_t5, &temp_t6, &temp_t7, &temp_t8,
					 &temp_t9, &temp_t10, &temp_t11, &temp_t12, &temp_t13, &temp_t14, &temp_t15, &temp_t16, &temp_t17, &temp_t18, &temp_t19, &temp_t20,&temp_t21, &temp_t22, &temp_t23, &temp_t24, 
					 &temp_t25, &temp_t26, &temp_t27, &temp_t28, &temp_t29, &temp_t30, &temp_t31);

				dot_prod10  = _mm256_maddubs_epi16 (r1,temp_t0);   dot_prod20 = _mm256_maddubs_epi16 (r2,temp_t0); 
				dot_prod11  = _mm256_maddubs_epi16 (r1,temp_t1);   dot_prod21 = _mm256_maddubs_epi16 (r2,temp_t1); 
				dot_prod12  = _mm256_maddubs_epi16 (r1,temp_t2);   dot_prod22 = _mm256_maddubs_epi16 (r2,temp_t2); 
				dot_prod13  = _mm256_maddubs_epi16 (r1,temp_t3);   dot_prod23 = _mm256_maddubs_epi16 (r2,temp_t3); 
				dot_prod14  = _mm256_maddubs_epi16 (r1,temp_t4);   dot_prod24 = _mm256_maddubs_epi16 (r2,temp_t4); 
				dot_prod15  = _mm256_maddubs_epi16 (r1,temp_t5);   dot_prod25 = _mm256_maddubs_epi16 (r2,temp_t5); 
				dot_prod16  = _mm256_maddubs_epi16 (r1,temp_t6);   dot_prod26 = _mm256_maddubs_epi16 (r2,temp_t6); 
				dot_prod17  = _mm256_maddubs_epi16 (r1,temp_t7);   dot_prod27 = _mm256_maddubs_epi16 (r2,temp_t7); 
				dot_prod18  = _mm256_maddubs_epi16 (r1,temp_t8);   dot_prod28 = _mm256_maddubs_epi16 (r2,temp_t8); 
				dot_prod19  = _mm256_maddubs_epi16 (r1,temp_t9);   dot_prod29 = _mm256_maddubs_epi16 (r2,temp_t9); 
				dot_prod110 = _mm256_maddubs_epi16 (r1,temp_t10);  dot_prod210 = _mm256_maddubs_epi16 (r2,temp_t10);
				dot_prod111 = _mm256_maddubs_epi16 (r1,temp_t11);  dot_prod211 = _mm256_maddubs_epi16 (r2,temp_t11);
				dot_prod112 = _mm256_maddubs_epi16 (r1,temp_t12);  dot_prod212 = _mm256_maddubs_epi16 (r2,temp_t12);
				dot_prod113 = _mm256_maddubs_epi16 (r1,temp_t13);  dot_prod213 = _mm256_maddubs_epi16 (r2,temp_t13);
				dot_prod114 = _mm256_maddubs_epi16 (r1,temp_t14);  dot_prod214 = _mm256_maddubs_epi16 (r2,temp_t14);
				dot_prod115 = _mm256_maddubs_epi16 (r1,temp_t15);  dot_prod215 = _mm256_maddubs_epi16 (r2,temp_t15);
				dot_prod116 = _mm256_maddubs_epi16 (r1,temp_t16);  dot_prod216 = _mm256_maddubs_epi16 (r2,temp_t16);
				dot_prod117 = _mm256_maddubs_epi16 (r1,temp_t17);  dot_prod217 = _mm256_maddubs_epi16 (r2,temp_t17);
				dot_prod118 = _mm256_maddubs_epi16 (r1,temp_t18);  dot_prod218 = _mm256_maddubs_epi16 (r2,temp_t18);
				dot_prod119 = _mm256_maddubs_epi16 (r1,temp_t19);  dot_prod219 = _mm256_maddubs_epi16 (r2,temp_t19);
				dot_prod120 = _mm256_maddubs_epi16 (r1,temp_t20);  dot_prod220 = _mm256_maddubs_epi16 (r2,temp_t20);
				dot_prod121 = _mm256_maddubs_epi16 (r1,temp_t21);  dot_prod221 = _mm256_maddubs_epi16 (r2,temp_t21);
				dot_prod122 = _mm256_maddubs_epi16 (r1,temp_t22);  dot_prod222 = _mm256_maddubs_epi16 (r2,temp_t22);
				dot_prod123 = _mm256_maddubs_epi16 (r1,temp_t23);  dot_prod223 = _mm256_maddubs_epi16 (r2,temp_t23);
				dot_prod124 = _mm256_maddubs_epi16 (r1,temp_t24);  dot_prod224 = _mm256_maddubs_epi16 (r2,temp_t24);
				dot_prod125 = _mm256_maddubs_epi16 (r1,temp_t25);  dot_prod225 = _mm256_maddubs_epi16 (r2,temp_t25);
				dot_prod126 = _mm256_maddubs_epi16 (r1,temp_t26);  dot_prod226 = _mm256_maddubs_epi16 (r2,temp_t26);
				dot_prod127 = _mm256_maddubs_epi16 (r1,temp_t27);  dot_prod227 = _mm256_maddubs_epi16 (r2,temp_t27);
				dot_prod128 = _mm256_maddubs_epi16 (r1,temp_t28);  dot_prod228 = _mm256_maddubs_epi16 (r2,temp_t28);
				dot_prod129 = _mm256_maddubs_epi16 (r1,temp_t29);  dot_prod229 = _mm256_maddubs_epi16 (r2,temp_t29);
				dot_prod120 = _mm256_maddubs_epi16 (r1,temp_t30);  dot_prod230 = _mm256_maddubs_epi16 (r2,temp_t30);
				dot_prod131 = _mm256_maddubs_epi16 (r1,temp_t31);  dot_prod231 = _mm256_maddubs_epi16 (r2,temp_t31);

				acc11_0  = _mm256_add_epi16(acc11_0,dot_prod10);   acc12_0  = _mm256_add_epi16(acc12_0,dot_prod11); 
				acc21_0  = _mm256_add_epi16(acc21_0,dot_prod20);   acc22_0  = _mm256_add_epi16(acc22_0,dot_prod21);
				acc11_1  = _mm256_add_epi16(acc11_1,dot_prod12);   acc12_1  = _mm256_add_epi16(acc12_1,dot_prod13); 
				acc21_1  = _mm256_add_epi16(acc21_1,dot_prod22);   acc22_1  = _mm256_add_epi16(acc22_1,dot_prod23);
				acc11_2  = _mm256_add_epi16(acc11_2,dot_prod14);   acc12_2  = _mm256_add_epi16(acc12_2,dot_prod15); 
				acc21_2  = _mm256_add_epi16(acc21_2,dot_prod24);   acc22_2  = _mm256_add_epi16(acc22_2,dot_prod25);
				acc11_3  = _mm256_add_epi16(acc11_3,dot_prod16);   acc12_3  = _mm256_add_epi16(acc12_3,dot_prod17); 
				acc21_3  = _mm256_add_epi16(acc21_3,dot_prod26);   acc22_3  = _mm256_add_epi16(acc22_3,dot_prod27);
				acc11_4  = _mm256_add_epi16(acc11_4,dot_prod18);   acc12_4  = _mm256_add_epi16(acc12_4,dot_prod19); 
				acc21_4  = _mm256_add_epi16(acc21_4,dot_prod28);   acc22_4  = _mm256_add_epi16(acc22_4,dot_prod29);
				acc11_5  = _mm256_add_epi16(acc11_5,dot_prod110);  acc12_5  = _mm256_add_epi16(acc12_5,dot_prod111); 
				acc21_5  = _mm256_add_epi16(acc21_5,dot_prod210);  acc22_5  = _mm256_add_epi16(acc22_5,dot_prod211);
				acc11_6  = _mm256_add_epi16(acc11_6,dot_prod112);  acc12_6  = _mm256_add_epi16(acc12_6,dot_prod113); 
				acc21_6  = _mm256_add_epi16(acc21_6,dot_prod212);  acc22_6  = _mm256_add_epi16(acc22_6,dot_prod213);
				acc11_7  = _mm256_add_epi16(acc11_7,dot_prod114);  acc12_7  = _mm256_add_epi16(acc12_7,dot_prod115); 
				acc21_7  = _mm256_add_epi16(acc21_7,dot_prod214);  acc22_7  = _mm256_add_epi16(acc22_7,dot_prod215);
				acc11_8  = _mm256_add_epi16(acc11_8,dot_prod116);  acc12_8  = _mm256_add_epi16(acc12_8,dot_prod117); 
				acc21_8  = _mm256_add_epi16(acc21_8,dot_prod216);  acc22_8  = _mm256_add_epi16(acc22_8,dot_prod217);
				acc11_9  = _mm256_add_epi16(acc11_9,dot_prod118);  acc12_9  = _mm256_add_epi16(acc12_9,dot_prod119); 
				acc21_9  = _mm256_add_epi16(acc21_9,dot_prod218);  acc22_9  = _mm256_add_epi16(acc22_9,dot_prod219);
				acc11_10 = _mm256_add_epi16(acc11_10,dot_prod120); acc12_10 = _mm256_add_epi16(acc12_10,dot_prod121); 
				acc21_10 = _mm256_add_epi16(acc21_10,dot_prod220); acc22_10 = _mm256_add_epi16(acc22_10,dot_prod221);
				acc11_11 = _mm256_add_epi16(acc11_11,dot_prod122); acc12_11 = _mm256_add_epi16(acc12_11,dot_prod123); 
				acc21_11 = _mm256_add_epi16(acc21_11,dot_prod222); acc22_11 = _mm256_add_epi16(acc22_11,dot_prod223);
				acc11_12 = _mm256_add_epi16(acc11_12,dot_prod124); acc12_12 = _mm256_add_epi16(acc12_12,dot_prod125); 
				acc21_12 = _mm256_add_epi16(acc21_12,dot_prod224); acc22_12 = _mm256_add_epi16(acc22_12,dot_prod225);
				acc11_13 = _mm256_add_epi16(acc11_13,dot_prod126); acc12_13 = _mm256_add_epi16(acc12_13,dot_prod127); 
				acc21_13 = _mm256_add_epi16(acc21_13,dot_prod226); acc22_13 = _mm256_add_epi16(acc22_13,dot_prod227);
				acc11_14 = _mm256_add_epi16(acc11_14,dot_prod128); acc12_14 = _mm256_add_epi16(acc12_14,dot_prod129); 
				acc21_14 = _mm256_add_epi16(acc21_14,dot_prod228); acc22_14 = _mm256_add_epi16(acc22_14,dot_prod229);
				acc11_15 = _mm256_add_epi16(acc11_15,dot_prod130); acc12_15 = _mm256_add_epi16(acc12_15,dot_prod131); 
				acc21_15 = _mm256_add_epi16(acc21_15,dot_prod230); acc22_15 = _mm256_add_epi16(acc22_15,dot_prod231);
				
			}

			acc11_int0  += _mm256_haddsi_epi16(acc11_0);  acc12_int0  += _mm256_haddsi_epi16(acc12_0);
			acc21_int0  += _mm256_haddsi_epi16(acc21_0);  acc22_int0  += _mm256_haddsi_epi16(acc22_0);
			acc11_int1  += _mm256_haddsi_epi16(acc11_1);  acc12_int1  += _mm256_haddsi_epi16(acc12_1);
			acc21_int1  += _mm256_haddsi_epi16(acc21_1);  acc22_int1  += _mm256_haddsi_epi16(acc22_1);
			acc11_int2  += _mm256_haddsi_epi16(acc11_2);  acc12_int2  += _mm256_haddsi_epi16(acc12_2);
			acc21_int2  += _mm256_haddsi_epi16(acc21_2);  acc22_int2  += _mm256_haddsi_epi16(acc22_2);
			acc11_int3  += _mm256_haddsi_epi16(acc11_3);  acc12_int3  += _mm256_haddsi_epi16(acc12_3);
			acc21_int3  += _mm256_haddsi_epi16(acc21_3);  acc22_int3  += _mm256_haddsi_epi16(acc22_3);
			acc11_int4  += _mm256_haddsi_epi16(acc11_4);  acc12_int4  += _mm256_haddsi_epi16(acc12_4);
			acc21_int4  += _mm256_haddsi_epi16(acc21_4);  acc22_int4  += _mm256_haddsi_epi16(acc22_4);
			acc11_int5  += _mm256_haddsi_epi16(acc11_5);  acc12_int5  += _mm256_haddsi_epi16(acc12_5);
			acc21_int5  += _mm256_haddsi_epi16(acc21_5);  acc22_int5  += _mm256_haddsi_epi16(acc22_5);
			acc11_int6  += _mm256_haddsi_epi16(acc11_6);  acc12_int6  += _mm256_haddsi_epi16(acc12_6);
			acc21_int6  += _mm256_haddsi_epi16(acc21_6);  acc22_int6  += _mm256_haddsi_epi16(acc22_6);
			acc11_int7  += _mm256_haddsi_epi16(acc11_7);  acc12_int7  += _mm256_haddsi_epi16(acc12_7);
			acc21_int7  += _mm256_haddsi_epi16(acc21_7);  acc22_int7  += _mm256_haddsi_epi16(acc22_7);
			acc11_int8  += _mm256_haddsi_epi16(acc11_8);  acc12_int8  += _mm256_haddsi_epi16(acc12_8);
			acc21_int8  += _mm256_haddsi_epi16(acc21_8);  acc22_int8  += _mm256_haddsi_epi16(acc22_8);
			acc11_int9  += _mm256_haddsi_epi16(acc11_9);  acc12_int9  += _mm256_haddsi_epi16(acc12_9);
			acc21_int9  += _mm256_haddsi_epi16(acc21_9);  acc22_int9  += _mm256_haddsi_epi16(acc22_9);
			acc11_int10 += _mm256_haddsi_epi16(acc11_10); acc12_int10 += _mm256_haddsi_epi16(acc12_10);
			acc21_int10 += _mm256_haddsi_epi16(acc21_10); acc22_int10 += _mm256_haddsi_epi16(acc22_10);
			acc11_int11 += _mm256_haddsi_epi16(acc11_11); acc12_int11 += _mm256_haddsi_epi16(acc12_11);
			acc21_int11 += _mm256_haddsi_epi16(acc21_11); acc22_int11 += _mm256_haddsi_epi16(acc22_11);
			acc11_int12 += _mm256_haddsi_epi16(acc11_12); acc12_int12 += _mm256_haddsi_epi16(acc12_12);
			acc21_int12 += _mm256_haddsi_epi16(acc21_12); acc22_int12 += _mm256_haddsi_epi16(acc22_12);
			acc11_int13 += _mm256_haddsi_epi16(acc11_13); acc12_int13 += _mm256_haddsi_epi16(acc12_13);
			acc21_int13 += _mm256_haddsi_epi16(acc21_13); acc22_int13 += _mm256_haddsi_epi16(acc22_13);
			acc11_int14 += _mm256_haddsi_epi16(acc11_14); acc12_int14 += _mm256_haddsi_epi16(acc12_14);
			acc21_int14 += _mm256_haddsi_epi16(acc21_14); acc22_int14 += _mm256_haddsi_epi16(acc22_14);
			acc11_int15 += _mm256_haddsi_epi16(acc11_15); acc12_int15 += _mm256_haddsi_epi16(acc12_15);
			acc21_int15 += _mm256_haddsi_epi16(acc21_15); acc22_int15 += _mm256_haddsi_epi16(acc22_15);

			for(int u=0;u<k; u = u+1){
					acc11_int0 += l_int_mat[i*k + u].i1*r_int_mat[u*m + j+0].i1 + l_int_mat[i*k + u].i2*r_int_mat[u*m + j].i3;
					acc12_int0 += l_int_mat[i*k + u].i1*r_int_mat[u*m + j+0].i2 + l_int_mat[i*k + u].i2*r_int_mat[u*m + j+0].i4;
					acc21_int0 += l_int_mat[i*k + u].i3*r_int_mat[u*m + j+0].i1 + l_int_mat[i*k + u].i4*r_int_mat[u*m + j].i3;
					acc22_int0 += l_int_mat[i*k + u].i3*r_int_mat[u*m + j+0].i2 + l_int_mat[i*k + u].i4*r_int_mat[u*m + j+0].i4;
			}
			for(int u=0;u<k; u = u+1){
					acc11_int1 += l_int_mat[i*k + u].i1*r_int_mat[u*m + j+1].i1 + l_int_mat[i*k + u].i2*r_int_mat[u*m + j].i3;
					acc12_int1 += l_int_mat[i*k + u].i1*r_int_mat[u*m + j+1].i2 + l_int_mat[i*k + u].i2*r_int_mat[u*m + j+1].i4;
					acc21_int1 += l_int_mat[i*k + u].i3*r_int_mat[u*m + j+1].i1 + l_int_mat[i*k + u].i4*r_int_mat[u*m + j].i3;
					acc22_int1 += l_int_mat[i*k + u].i3*r_int_mat[u*m + j+1].i2 + l_int_mat[i*k + u].i4*r_int_mat[u*m + j+1].i4;
			}
			for(int u=0;u<k; u = u+1){
					acc11_int2 += l_int_mat[i*k + u].i1*r_int_mat[u*m + j+2].i1 + l_int_mat[i*k + u].i2*r_int_mat[u*m + j].i3;
					acc12_int2 += l_int_mat[i*k + u].i1*r_int_mat[u*m + j+2].i2 + l_int_mat[i*k + u].i2*r_int_mat[u*m + j+2].i4;
					acc21_int2 += l_int_mat[i*k + u].i3*r_int_mat[u*m + j+2].i1 + l_int_mat[i*k + u].i4*r_int_mat[u*m + j].i3;
					acc22_int2 += l_int_mat[i*k + u].i3*r_int_mat[u*m + j+2].i2 + l_int_mat[i*k + u].i4*r_int_mat[u*m + j+2].i4;
			}
			for(int u=0;u<k; u = u+1){
					acc11_int3 += l_int_mat[i*k + u].i1*r_int_mat[u*m + j+3].i1 + l_int_mat[i*k + u].i2*r_int_mat[u*m + j].i3;
					acc12_int3 += l_int_mat[i*k + u].i1*r_int_mat[u*m + j+3].i2 + l_int_mat[i*k + u].i2*r_int_mat[u*m + j+3].i4;
					acc21_int3 += l_int_mat[i*k + u].i3*r_int_mat[u*m + j+3].i1 + l_int_mat[i*k + u].i4*r_int_mat[u*m + j].i3;
					acc22_int3 += l_int_mat[i*k + u].i3*r_int_mat[u*m + j+3].i2 + l_int_mat[i*k + u].i4*r_int_mat[u*m + j+3].i4;
			}
			for(int u=0;u<k; u = u+1){
					acc11_int4 += l_int_mat[i*k + u].i1*r_int_mat[u*m + j+4].i1 + l_int_mat[i*k + u].i2*r_int_mat[u*m + j].i3;
					acc12_int4 += l_int_mat[i*k + u].i1*r_int_mat[u*m + j+4].i2 + l_int_mat[i*k + u].i2*r_int_mat[u*m + j+4].i4;
					acc21_int4 += l_int_mat[i*k + u].i3*r_int_mat[u*m + j+4].i1 + l_int_mat[i*k + u].i4*r_int_mat[u*m + j].i3;
					acc22_int4 += l_int_mat[i*k + u].i3*r_int_mat[u*m + j+4].i2 + l_int_mat[i*k + u].i4*r_int_mat[u*m + j+4].i4;
			}
			for(int u=0;u<k; u = u+1){
					acc11_int5 += l_int_mat[i*k + u].i1*r_int_mat[u*m + j+5].i1 + l_int_mat[i*k + u].i2*r_int_mat[u*m + j].i3;
					acc12_int5 += l_int_mat[i*k + u].i1*r_int_mat[u*m + j+5].i2 + l_int_mat[i*k + u].i2*r_int_mat[u*m + j+5].i4;
					acc21_int5 += l_int_mat[i*k + u].i3*r_int_mat[u*m + j+5].i1 + l_int_mat[i*k + u].i4*r_int_mat[u*m + j].i3;
					acc22_int5 += l_int_mat[i*k + u].i3*r_int_mat[u*m + j+5].i2 + l_int_mat[i*k + u].i4*r_int_mat[u*m + j+5].i4;
			}
			for(int u=0;u<k; u = u+1){
					acc11_int6 += l_int_mat[i*k + u].i1*r_int_mat[u*m + j+6].i1 + l_int_mat[i*k + u].i2*r_int_mat[u*m + j].i3;
					acc12_int6 += l_int_mat[i*k + u].i1*r_int_mat[u*m + j+6].i2 + l_int_mat[i*k + u].i2*r_int_mat[u*m + j+6].i4;
					acc21_int6 += l_int_mat[i*k + u].i3*r_int_mat[u*m + j+6].i1 + l_int_mat[i*k + u].i4*r_int_mat[u*m + j].i3;
					acc22_int6 += l_int_mat[i*k + u].i3*r_int_mat[u*m + j+6].i2 + l_int_mat[i*k + u].i4*r_int_mat[u*m + j+6].i4;
			}
			for(int u=0;u<k; u = u+1){
					acc11_int7 += l_int_mat[i*k + u].i1*r_int_mat[u*m + j+7].i1 + l_int_mat[i*k + u].i2*r_int_mat[u*m + j].i3;
					acc12_int7 += l_int_mat[i*k + u].i1*r_int_mat[u*m + j+7].i2 + l_int_mat[i*k + u].i2*r_int_mat[u*m + j+7].i4;
					acc21_int7 += l_int_mat[i*k + u].i3*r_int_mat[u*m + j+7].i1 + l_int_mat[i*k + u].i4*r_int_mat[u*m + j].i3;
					acc22_int7 += l_int_mat[i*k + u].i3*r_int_mat[u*m + j+7].i2 + l_int_mat[i*k + u].i4*r_int_mat[u*m + j+7].i4;
			}
			for(int u=0;u<k; u = u+1){
					acc11_int8 += l_int_mat[i*k + u].i1*r_int_mat[u*m + j+8].i1 + l_int_mat[i*k + u].i2*r_int_mat[u*m + j].i3;
					acc12_int8 += l_int_mat[i*k + u].i1*r_int_mat[u*m + j+8].i2 + l_int_mat[i*k + u].i2*r_int_mat[u*m + j+8].i4;
					acc21_int8 += l_int_mat[i*k + u].i3*r_int_mat[u*m + j+8].i1 + l_int_mat[i*k + u].i4*r_int_mat[u*m + j].i3;
					acc22_int8 += l_int_mat[i*k + u].i3*r_int_mat[u*m + j+8].i2 + l_int_mat[i*k + u].i4*r_int_mat[u*m + j+8].i4;
			}
			for(int u=0;u<k; u = u+1){
					acc11_int9 += l_int_mat[i*k + u].i1*r_int_mat[u*m + j+9].i1 + l_int_mat[i*k + u].i2*r_int_mat[u*m + j].i3;
					acc12_int9 += l_int_mat[i*k + u].i1*r_int_mat[u*m + j+9].i2 + l_int_mat[i*k + u].i2*r_int_mat[u*m + j+9].i4;
					acc21_int9 += l_int_mat[i*k + u].i3*r_int_mat[u*m + j+9].i1 + l_int_mat[i*k + u].i4*r_int_mat[u*m + j].i3;
					acc22_int9 += l_int_mat[i*k + u].i3*r_int_mat[u*m + j+9].i2 + l_int_mat[i*k + u].i4*r_int_mat[u*m + j+9].i4;
			}
			for(int u=0;u<k; u = u+1){
					acc11_int10 += l_int_mat[i*k + u].i1*r_int_mat[u*m + j+10].i1 + l_int_mat[i*k + u].i2*r_int_mat[u*m + j].i3;
					acc12_int10 += l_int_mat[i*k + u].i1*r_int_mat[u*m + j+10].i2 + l_int_mat[i*k + u].i2*r_int_mat[u*m + j+10].i4;
					acc21_int10 += l_int_mat[i*k + u].i3*r_int_mat[u*m + j+10].i1 + l_int_mat[i*k + u].i4*r_int_mat[u*m + j].i3;
					acc22_int10 += l_int_mat[i*k + u].i3*r_int_mat[u*m + j+10].i2 + l_int_mat[i*k + u].i4*r_int_mat[u*m + j+10].i4;
			}
			for(int u=0;u<k; u = u+1){
					acc11_int11 += l_int_mat[i*k + u].i1*r_int_mat[u*m + j+11].i1 + l_int_mat[i*k + u].i2*r_int_mat[u*m + j].i3;
					acc12_int11 += l_int_mat[i*k + u].i1*r_int_mat[u*m + j+11].i2 + l_int_mat[i*k + u].i2*r_int_mat[u*m + j+11].i4;
					acc21_int11 += l_int_mat[i*k + u].i3*r_int_mat[u*m + j+11].i1 + l_int_mat[i*k + u].i4*r_int_mat[u*m + j].i3;
					acc22_int11 += l_int_mat[i*k + u].i3*r_int_mat[u*m + j+11].i2 + l_int_mat[i*k + u].i4*r_int_mat[u*m + j+11].i4;
			}
			for(int u=0;u<k; u = u+1){
					acc11_int12 += l_int_mat[i*k + u].i1*r_int_mat[u*m + j+12].i1 + l_int_mat[i*k + u].i2*r_int_mat[u*m + j].i3;
					acc12_int12 += l_int_mat[i*k + u].i1*r_int_mat[u*m + j+12].i2 + l_int_mat[i*k + u].i2*r_int_mat[u*m + j+12].i4;
					acc21_int12 += l_int_mat[i*k + u].i3*r_int_mat[u*m + j+12].i1 + l_int_mat[i*k + u].i4*r_int_mat[u*m + j].i3;
					acc22_int12 += l_int_mat[i*k + u].i3*r_int_mat[u*m + j+12].i2 + l_int_mat[i*k + u].i4*r_int_mat[u*m + j+12].i4;
			}
			for(int u=0;u<k; u = u+1){
					acc11_int13 += l_int_mat[i*k + u].i1*r_int_mat[u*m + j+13].i1 + l_int_mat[i*k + u].i2*r_int_mat[u*m + j].i3;
					acc12_int13 += l_int_mat[i*k + u].i1*r_int_mat[u*m + j+13].i2 + l_int_mat[i*k + u].i2*r_int_mat[u*m + j+13].i4;
					acc21_int13 += l_int_mat[i*k + u].i3*r_int_mat[u*m + j+13].i1 + l_int_mat[i*k + u].i4*r_int_mat[u*m + j].i3;
					acc22_int13 += l_int_mat[i*k + u].i3*r_int_mat[u*m + j+13].i2 + l_int_mat[i*k + u].i4*r_int_mat[u*m + j+13].i4;
			}
			for(int u=0;u<k; u = u+1){
					acc11_int14 += l_int_mat[i*k + u].i1*r_int_mat[u*m + j+14].i1 + l_int_mat[i*k + u].i2*r_int_mat[u*m + j].i3;
					acc12_int14 += l_int_mat[i*k + u].i1*r_int_mat[u*m + j+14].i2 + l_int_mat[i*k + u].i2*r_int_mat[u*m + j+14].i4;
					acc21_int14 += l_int_mat[i*k + u].i3*r_int_mat[u*m + j+14].i1 + l_int_mat[i*k + u].i4*r_int_mat[u*m + j].i3;
					acc22_int14 += l_int_mat[i*k + u].i3*r_int_mat[u*m + j+14].i2 + l_int_mat[i*k + u].i4*r_int_mat[u*m + j+14].i4;
			}
			for(int u=0;u<k; u = u+1){
					acc11_int15 += l_int_mat[i*k + u].i1*r_int_mat[u*m + j+15].i1 + l_int_mat[i*k + u].i2*r_int_mat[u*m + j].i3;
					acc12_int15 += l_int_mat[i*k + u].i1*r_int_mat[u*m + j+15].i2 + l_int_mat[i*k + u].i2*r_int_mat[u*m + j+15].i4;
					acc21_int15 += l_int_mat[i*k + u].i3*r_int_mat[u*m + j+15].i1 + l_int_mat[i*k + u].i4*r_int_mat[u*m + j].i3;
					acc22_int15 += l_int_mat[i*k + u].i3*r_int_mat[u*m + j+15].i2 + l_int_mat[i*k + u].i4*r_int_mat[u*m + j+15].i4;
			}

			acc[(2*i)*2*m + 2*j+4*0] = acc11_int0; 	    acc[(2*i)*2*m + 2*j+4*0+1] = acc12_int0;    acc[(2*i+1)*2*m + 2*j+4*0] = acc21_int0;    acc[(2*i+1)*2*m + 2*j+4*0+1] = acc22_int0;
			acc[(2*i)*2*m + 2*j+4*0+2] = acc11_int1;	acc[(2*i)*2*m + 2*j+4*0+3] = acc12_int1;	acc[(2*i+1)*2*m + 2*j+4*0+2] = acc21_int1;  acc[(2*i+1)*2*m + 2*j+4*0+3] = acc22_int1;
			acc[(2*i)*2*m + 2*j+4*1] = acc11_int2; 	    acc[(2*i)*2*m + 2*j+4*1+1] = acc12_int2;    acc[(2*i+1)*2*m + 2*j+4*1] = acc21_int2;    acc[(2*i+1)*2*m + 2*j+4*1+1] = acc22_int2;
			acc[(2*i)*2*m + 2*j+4*1+2] = acc11_int3;	acc[(2*i)*2*m + 2*j+4*1+3] = acc12_int3;	acc[(2*i+1)*2*m + 2*j+4*1+2] = acc21_int3;  acc[(2*i+1)*2*m + 2*j+4*1+3] = acc22_int3;
			acc[(2*i)*2*m + 2*j+4*2] = acc11_int4; 	    acc[(2*i)*2*m + 2*j+4*2+1] = acc12_int4;    acc[(2*i+1)*2*m + 2*j+4*2] = acc21_int4;    acc[(2*i+1)*2*m + 2*j+4*2+1] = acc22_int4;
			acc[(2*i)*2*m + 2*j+4*2+2] = acc11_int5;	acc[(2*i)*2*m + 2*j+4*2+3] = acc12_int5;	acc[(2*i+1)*2*m + 2*j+4*2+2] = acc21_int5;  acc[(2*i+1)*2*m + 2*j+4*2+3] = acc22_int5;
			acc[(2*i)*2*m + 2*j+4*3] = acc11_int6; 	    acc[(2*i)*2*m + 2*j+4*3+1] = acc12_int6;    acc[(2*i+1)*2*m + 2*j+4*3] = acc21_int6;    acc[(2*i+1)*2*m + 2*j+4*3+1] = acc22_int6;
			acc[(2*i)*2*m + 2*j+4*3+2] = acc11_int7;	acc[(2*i)*2*m + 2*j+4*3+3] = acc12_int7;	acc[(2*i+1)*2*m + 2*j+4*3+2] = acc21_int7;  acc[(2*i+1)*2*m + 2*j+4*3+3] = acc22_int7;
			acc[(2*i)*2*m + 2*j+4*4] = acc11_int8; 	    acc[(2*i)*2*m + 2*j+4*4+1] = acc12_int8;    acc[(2*i+1)*2*m + 2*j+4*4] = acc21_int8;    acc[(2*i+1)*2*m + 2*j+4*4+1] = acc22_int8;
			acc[(2*i)*2*m + 2*j+4*4+2] = acc11_int9;	acc[(2*i)*2*m + 2*j+4*4+3] = acc12_int9;	acc[(2*i+1)*2*m + 2*j+4*4+2] = acc21_int9;  acc[(2*i+1)*2*m + 2*j+4*4+3] = acc22_int9;
			acc[(2*i)*2*m + 2*j+4*5] = acc11_int10; 	acc[(2*i)*2*m + 2*j+4*5+1] = acc12_int10;   acc[(2*i+1)*2*m + 2*j+4*5] = acc21_int10;   acc[(2*i+1)*2*m + 2*j+4*5+1] = acc22_int10;
			acc[(2*i)*2*m + 2*j+4*5+2] = acc11_int11;	acc[(2*i)*2*m + 2*j+4*5+3] = acc12_int11;	acc[(2*i+1)*2*m + 2*j+4*5+2] = acc21_int11; acc[(2*i+1)*2*m + 2*j+4*5+3] = acc22_int11;
			acc[(2*i)*2*m + 2*j+4*6] = acc11_int12; 	acc[(2*i)*2*m + 2*j+4*6+1] = acc12_int12;   acc[(2*i+1)*2*m + 2*j+4*6] = acc21_int12;   acc[(2*i+1)*2*m + 2*j+4*6+1] = acc22_int12;
			acc[(2*i)*2*m + 2*j+4*6+2] = acc11_int13;	acc[(2*i)*2*m + 2*j+4*6+3] = acc12_int13;	acc[(2*i+1)*2*m + 2*j+4*6+2] = acc21_int13; acc[(2*i+1)*2*m + 2*j+4*6+3] = acc22_int13;
			acc[(2*i)*2*m + 2*j+4*7] = acc11_int14; 	acc[(2*i)*2*m + 2*j+4*7+1] = acc12_int14;   acc[(2*i+1)*2*m + 2*j+4*7] = acc21_int14;   acc[(2*i+1)*2*m + 2*j+4*7+1] = acc22_int14;
			acc[(2*i)*2*m + 2*j+4*7+2] = acc11_int15;	acc[(2*i)*2*m + 2*j+4*7+3] = acc12_int15;	acc[(2*i+1)*2*m + 2*j+4*7+2] = acc21_int15; acc[(2*i+1)*2*m + 2*j+4*7+3] = acc22_int15;
		}

		for(; j<m; j = j+1){
			acc1 = 0;
			acc2 = 0;
			acc3 = 0;
			acc4 = 0;
			for (int t=0; t<k; t = t+1){
				uint4x4_t l_element = l_int_mat[i*k + t]; 
				uint4x4_t r_elment = r_int_mat[t*m + j];
				acc1 += (l_element.i1) * (r_elment.i1) + 
						(l_element.i2) * (r_elment.i3);

				acc2 += (l_element.i1) * (r_elment.i2) + 
						(l_element.i2) * (r_elment.i4);

				acc3 += (l_element.i3) * (r_elment.i1) + 
						(l_element.i4) * (r_elment.i3);

				acc4 += (l_element.i3) * (r_elment.i2) + 
						(l_element.i4) * (r_elment.i4);

			}

			acc[(2*i)*2*m + 2*j] = acc1;
			acc[(2*i)*2*m + 2*j+1] = acc2;
			acc[(2*i+1)*2*m + 2*j] = acc3;
			acc[(2*i+1)*2*m + 2*j+1] = acc4;
			
		}
	}	
}