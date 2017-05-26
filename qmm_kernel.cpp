#include <stdio.h>
#include <stdint.h>
#include "utils.h"
#include "qmm_kernel.h"
#include <math.h>
#include <stdlib.h>
#include <immintrin.h>

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

			for ( t=0; t<(k-31); t = t+32){
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

			for (int u = 0; i < 8; u+=1){
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