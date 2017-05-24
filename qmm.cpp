#include <stdio.h>
#include <stdint.h>
#include "utils.h"
#include "qmm.h"
#include <math.h>
#include <stdlib.h>
#include <immintrin.h> 


// Implementing the MM with struct the require 8 bits but contain only bits of info
void qmm_space_waste(float l_scale, float r_scale, float result_scale, uint4x1_t l_offset, uint4x1_t r_offset, 
	uint4x1_t result_offset, uint4x1_t* l_int_mat, uint4x1_t* r_int_mat, uint4x1_t* result_int_mat, 
	int n, int k, int m){

	int16_t accumulator;

	for(int i=0;i<n;i++){
		for(int j=0;j<m;j++){
			accumulator = 0;
			for(int t=0; t<k;t++){
				accumulator += (l_int_mat[i*k + t].i - l_offset.i) * (r_int_mat[t*m + j].i - r_offset.i);
			}
		result_int_mat[i*m+j].i = saturate(round(result_offset.i + (l_scale * r_scale/result_scale) * accumulator));
		}
	}
}


void qmm_naive(float l_scale, float r_scale, float result_scale, uint4x4_t l_offset, uint4x4_t r_offset, 
	uint4x4_t result_offset, uint4x4_t* l_int_mat, uint4x4_t* r_int_mat, uint4x4_t* result_int_mat, 
	int n, int k, int m){
	/*
	Every loop iteration makes us go through one uint4x4_t. The difference with the previous csae is that this
	contains 4 ints now and not one.
	*/
	int16_t acc1, acc2, acc3, acc4;

	// Internally we divide everything by 2 because one uint4x4_t contains 4 integers of 4 bit organize in 2 cols and 2 rows

	n = n/2;
	m = m/2;
	k = k/2;

	for(int i=0; i<n; i = i+1){
		for(int j=0; j<m; j = j+1){
			acc1 = 0;
			acc2 = 0;
			acc3 = 0;
			acc4 = 0;
			for (int t=0; t<k; t = t+1){
				acc1 += (l_int_mat[i*k + t].i1 - l_offset.i1) * (r_int_mat[t*m + j].i1 - r_offset.i1) + 
						(l_int_mat[i*k + t].i2 - l_offset.i1) * (r_int_mat[t*m + j].i3 - r_offset.i1);

				acc2 += (l_int_mat[i*k + t].i1 - l_offset.i1) * (r_int_mat[t*m + j].i2 - r_offset.i1) + 
						(l_int_mat[i*k + t].i2 - l_offset.i1) * (r_int_mat[t*m + j].i4 - r_offset.i1);

				acc3 += (l_int_mat[i*k + t].i3 - l_offset.i1) * (r_int_mat[t*m + j].i1 - r_offset.i1) + 
						(l_int_mat[i*k + t].i4 - l_offset.i1) * (r_int_mat[t*m + j].i3 - r_offset.i1);

				acc4 += (l_int_mat[i*k + t].i3 - l_offset.i1) * (r_int_mat[t*m + j].i2 - r_offset.i1) + 
						(l_int_mat[i*k + t].i4 - l_offset.i1) * (r_int_mat[t*m + j].i4 - r_offset.i1);

			}
			
		result_int_mat[i*m + j].i1 = saturate(round(result_offset.i1 + (l_scale * r_scale/result_scale) * acc1));
		result_int_mat[i*m + j].i2 = saturate(round(result_offset.i1 + (l_scale * r_scale/result_scale) * acc2));
		result_int_mat[i*m + j].i3 = saturate(round(result_offset.i1 + (l_scale * r_scale/result_scale) * acc3));
		result_int_mat[i*m + j].i4 = saturate(round(result_offset.i1 + (l_scale * r_scale/result_scale) * acc4));
		
		}
	}
}



void qmm_trick(float l_scale, float r_scale, float result_scale, uint4x4_t l_offset, uint4x4_t r_offset, 
	uint4x4_t result_offset, uint4x4_t* l_int_mat, uint4x4_t* r_int_mat, uint4x4_t* result_int_mat, 
	int n, int k, int m){

	int16_t acc1, acc2, acc3, acc4;

	// Internally we divide everything by 2 because one uint4x4_t contains 4 integers of 4 bit organize in 2 cols and 2 rows

	n = n/2;
	m = m/2;
	k = k/2;


	// Precompute additional terms
	uint16_t* term2 = (uint16_t*)malloc(sizeof(uint16_t)* m *2);
	uint16_t* term3 = (uint16_t*)malloc(sizeof(uint16_t)* n *2);
	uint16_t term4;
	uint4x4_t r_elment,l_element;
	uint16_t sum1, sum2;

	// Sum over the entries of each col of rhs and multiply by left offset 
	for(int j =0; j<m; j+=1){
		sum1 = 0;
		sum2 = 0;
		for(int i = 0; i<k; i++){
			r_elment = r_int_mat[i*m + j]; 
			sum1 += r_elment.i1 + r_elment.i3;
			sum2 += r_elment.i2 + r_elment.i4;
		}
		term2[2*j] = l_offset.i1 * sum1;
		term2[2*j + 1] = l_offset.i1 * sum2;	
	}

	// Sum over the entries of each row of lhs and multiply by right offset
	for (int i=0; i<n; i++){
		sum1 = 0;
		sum2 = 0;
		for (int j=0; j<k; j++){
			l_element = l_int_mat[i*k + j];
			sum1 += l_element.i1 + l_element.i2;
			sum2 += l_element.i3 + l_element.i4;
		}
		term3[2*i] = r_offset.i1 * sum1;
		term3[2*i + 1] = r_offset.i1 * sum2;
	}

	term4 = l_offset.i1 * r_offset.i1 * (k * 2);
	

	for(int i=0; i<n; i = i+1){
		for(int j=0; j<m; j = j+1){
			acc1 = 0;
			acc2 = 0;
			acc3 = 0;
			acc4 = 0;
			for (int t=0; t<k; t = t+1){
				l_element = l_int_mat[i*k + t]; 
				r_elment = r_int_mat[t*m + j];
				acc1 += (l_element.i1) * (r_elment.i1) + 
						(l_element.i2) * (r_elment.i3);

				acc2 += (l_element.i1) * (r_elment.i2) + 
						(l_element.i2) * (r_elment.i4);

				acc3 += (l_element.i3) * (r_elment.i1) + 
						(l_element.i4) * (r_elment.i3);

				acc4 += (l_element.i3) * (r_elment.i2) + 
						(l_element.i4) * (r_elment.i4);

			}
			
		acc1 = acc1 - term2[2*j] - term3[2*i] + term4;
		acc2 = acc2 - term2[2*j + 1] - term3[2*i] + term4;
		acc3 = acc3 - term2[2*j] - term3[2*i + 1] + term4;
		acc4 = acc4 - term2[2*j + 1] - term3[2*i + 1] + term4;

		result_int_mat[i*m + j].i1 = saturate(round(result_offset.i1 + (l_scale * r_scale/result_scale) * acc1));
		result_int_mat[i*m + j].i2 = saturate(round(result_offset.i1 + (l_scale * r_scale/result_scale) * acc2));
		result_int_mat[i*m + j].i3 = saturate(round(result_offset.i1 + (l_scale * r_scale/result_scale) * acc3));
		result_int_mat[i*m + j].i4 = saturate(round(result_offset.i1 + (l_scale * r_scale/result_scale) * acc4));
		
		}
	}

}



void qmm_trick_blocking(float l_scale, float r_scale, float result_scale, uint4x4_t l_offset, uint4x4_t r_offset, 
	uint4x4_t result_offset, uint4x4_t* l_int_mat, uint4x4_t* r_int_mat, uint4x4_t* result_int_mat, 
	int n, int k, int m){

	int16_t acc1, acc2, acc3, acc4;
	int Nb=30;
	int i,j,t;
	// Internally we divide everything by 2 because one uint4x4_t contains 4 integers of 4 bit organize in 2 cols and 2 rows

	n = n/2;
	m = m/2;
	k = k/2;


	// Precompute additional terms
	uint16_t* term2 = (uint16_t*)malloc(sizeof(uint16_t)* m *2);
	uint16_t* term3 = (uint16_t*)malloc(sizeof(uint16_t)* n *2);
	uint16_t term4,t001,t002,t003,t004,t101,t102,t103,t104,t011,t012,t013,t014,t021,t022,t023,t024,t111,t112,t113,t114,t121,t122,t123,t124,t201,t202,t203,t204;
	uint16_t t211,t212,t213,t214,t221,t222,t223,t224;
	uint4x4_t r_elment,l_element,c00,c01,c02,c10,c11,c12,c20,c21,c22,l0,l1,l2,r0,r1,r2;
	uint16_t sum1, sum2;
	uint16_t* acc = (uint16_t*)malloc(sizeof(uint16_t)*2* n *2*m);

	for (int i = 0; i < 4*n*m; ++i)
	{
		acc[i]=0;
	}

	// Sum over the entries of each col of rhs and multiply by left offset 
	for(int j =0; j<m; j+=1){
		sum1 = 0;
		sum2 = 0;
		for(int i = 0; i<k; i++){
			r_elment = r_int_mat[i*m + j]; 
			sum1 += r_elment.i1 + r_elment.i3;
			sum2 += r_elment.i2 + r_elment.i4;
		}
		term2[2*j] = l_offset.i1 * sum1;
		term2[2*j + 1] = l_offset.i1 * sum2;	
	}

	// Sum over the entries of each row of lhs and multiply by right offset
	for (int i=0; i<n; i++){
		sum1 = 0;
		sum2 = 0;
		for (int j=0; j<k; j++){
			l_element = l_int_mat[i*k + j];
			sum1 += l_element.i1 + l_element.i2;
			sum2 += l_element.i3 + l_element.i4;
		}
		term3[2*i] = r_offset.i1 * sum1;
		term3[2*i + 1] = r_offset.i1 * sum2;
	}

	term4 = l_offset.i1 * r_offset.i1 * (k * 2);

	for(i=0; i<n-Nb; i +=Nb){
		for(j=0; j<m-Nb; j += Nb){
			for (t=0; t<k-Nb; t +=Nb){
				for(int i_p = i;i_p<i+Nb;i_p += 3){
					for(int j_p=j;j_p<j+Nb;j_p += 3){
						//loads of the accumulator matrix
						t001 = acc[2*i_p*m + 2*j_p];
						t002 = acc[2*i_p*m + 2*j_p+1];
						t003 = acc[(2*i_p+1)*m + 2*j_p];
						t004 = acc[(2*i_p+1)*m + 2*j_p+1];

						t011 = acc[2*i_p*m + 2*(j_p+1)];
						t012 = acc[2*i_p*m + 2*(j_p+1)+1];
						t013 = acc[(2*i_p+1)*m + 2*(j_p+1)];
						t014 = acc[(2*i_p+1)*m + 2*(j_p+1)+1];

						t021 = acc[2*i_p*m + 2*(j_p+2)];
						t022 = acc[2*i_p*m + 2*(j_p+2)+1];
						t023 = acc[(2*i_p+1)*m + 2*(j_p+2)];
						t024 = acc[(2*i_p+1)*m + 2*(j_p+2)+1];

						t101 = acc[(2*(i_p+1))*m + 2*j_p];
						t102 = acc[(2*(i_p+1))*m + 2*j_p+1];
						t103 = acc[(2*(i_p+1)+1)*m + 2*j_p];
						t104 = acc[(2*(i_p+1)+1)*m + 2*j_p+1];

						t111 = acc[(2*(i_p+1))*m + 2*(j_p+1)];
						t112 = acc[(2*(i_p+1))*m + 2*(j_p+1)+1];
						t113 = acc[(2*(i_p+1)+1)*m + 2*(j_p+1)];
						t114 = acc[(2*(i_p+1)+1)*m + 2*(j_p+1)+1];

						t121 = acc[(2*(i_p+1))*m + 2*(j_p+2)];
						t122 = acc[(2*(i_p+1))*m + 2*(j_p+2)+1];
						t123 = acc[(2*(i_p+1)+1)*m + 2*(j_p+2)];
						t124 = acc[(2*(i_p+1)+1)*m + 2*(j_p+2)+1];

						t201 = acc[(2*(i_p+2))*m + 2*(j_p)];
						t202 = acc[(2*(i_p+2))*m + 2*(j_p)+1];
						t203 = acc[(2*(i_p+2)+1)*m + 2*(j_p)];
						t204 = acc[(2*(i_p+2)+1)*m + 2*(j_p)+1];

						t211 = acc[(2*(i_p+2))*m + 2*(j_p+1)];
						t212 = acc[(2*(i_p+2))*m + 2*(j_p+1)+1];
						t213 = acc[(2*(i_p+2)+1)*m + 2*(j_p+1)];
						t214 = acc[(2*(i_p+2)+1)*m + 2*(j_p+1)+1];

						t221 = acc[(2*(i_p+2))*m + 2*(j_p+2)];
						t222 = acc[(2*(i_p+2))*m + 2*(j_p+2)+1];
						t223 = acc[(2*(i_p+2)+1)*m + 2*(j_p+2)];
						t224 = acc[(2*(i_p+2)+1)*m + 2*(j_p+2)+1];

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
						acc[2*i_p*m + 2*j_p] = t001;
						acc[2*i_p*m + 2*j_p+1] = t002;
						acc[(2*i_p+1)*m + 2*j_p] = t003;
						acc[(2*i_p+1)*m + 2*j_p+1] = t004;

						acc[2*i_p*m + 2*(j_p+1)] = t011;
						acc[2*i_p*m + 2*(j_p+1)+1] = t012;
						acc[(2*i_p+1)*m + 2*(j_p+1)] = t013;
						acc[(2*i_p+1)*m + 2*(j_p+1)+1] = t014;

						acc[2*i_p*m + 2*(j_p+2)]  = t021;
						acc[2*i_p*m + 2*(j_p+2)+1] = t022;
						acc[(2*i_p+1)*m + 2*(j_p+2)] = t023;
						acc[(2*i_p+1)*m + 2*(j_p+2)+1] = t024;

						acc[(2*(i_p+1))*m + 2*j_p] = t101;
						acc[(2*(i_p+1))*m + 2*j_p+1] = t102;
					    acc[(2*(i_p+1)+1)*m + 2*j_p] = t103;
						acc[(2*(i_p+1)+1)*m + 2*j_p+1] = t104;

						acc[(2*(i_p+1))*m + 2*(j_p+1)] = t111;
						acc[(2*(i_p+1))*m + 2*(j_p+1)+1] = t112;
						acc[(2*(i_p+1)+1)*m + 2*(j_p+1)] = t113;
						acc[(2*(i_p+1)+1)*m + 2*(j_p+1)+1] = t114;

						acc[(2*(i_p+1))*m + 2*(j_p+2)] = t121;
						acc[(2*(i_p+1))*m + 2*(j_p+2)+1] = t122;
						acc[(2*(i_p+1)+1)*m + 2*(j_p+2)] = t123;
						acc[(2*(i_p+1)+1)*m + 2*(j_p+2)+1] = t124;

						acc[(2*(i_p+2))*m + 2*(j_p)] = t201;
						acc[(2*(i_p+2))*m + 2*(j_p)+1] = t202;
						acc[(2*(i_p+2)+1)*m + 2*(j_p)] = t203;
						acc[(2*(i_p+2)+1)*m + 2*(j_p)+1] = t204;

						acc[(2*(i_p+2))*m + 2*(j_p+1)] = t211;
						acc[(2*(i_p+2))*m + 2*(j_p+1)+1] = t212;
						acc[(2*(i_p+2)+1)*m + 2*(j_p+1)] = t213;
						acc[(2*(i_p+2)+1)*m + 2*(j_p+1)+1] = t214;

						acc[(2*(i_p+2))*m + 2*(j_p+2)] = t221;
						acc[(2*(i_p+2))*m + 2*(j_p+2)+1] = t222;
						acc[(2*(i_p+2)+1)*m + 2*(j_p+2)] = t223;
						acc[(2*(i_p+2)+1)*m + 2*(j_p+2)+1] = t224;

					}
				}
			}

			for (; t<k; t++ ){
				acc[2*i*m + 2*j] += l_int_mat[i*k + t].i1*r_int_mat[t*m + j].i1 + l_int_mat[i*k + t].i2*r_int_mat[t*m + j].i3;
				acc[2*i*m + 2*j+1] += l_int_mat[i*k + t].i1*r_int_mat[t*m + j].i2 + l_int_mat[i*k + t].i2*r_int_mat[t*m + j].i4;
				acc[(2*i+1)*m + 2*j] += l_int_mat[i*k + t].i3*r_int_mat[t*m + j].i1 + l_int_mat[i*k + t].i4*r_int_mat[t*m + j].i3;
				acc[(2*i+1)*m + 2*j+1] += l_int_mat[i*k + t].i3*r_int_mat[t*m + j].i2 + l_int_mat[i*k + t].i4*r_int_mat[t*m + j].i4;
			}
		}

		for (; j < m; j++)
		{
			for (t=0; t<k; t++ ){
				acc[2*i*m + 2*j] += l_int_mat[i*k + t].i1*r_int_mat[t*m + j].i1 + l_int_mat[i*k + t].i2*r_int_mat[t*m + j].i3;
				acc[2*i*m + 2*j+1] += l_int_mat[i*k + t].i1*r_int_mat[t*m + j].i2 + l_int_mat[i*k + t].i2*r_int_mat[t*m + j].i4;
				acc[(2*i+1)*m + 2*j] += l_int_mat[i*k + t].i3*r_int_mat[t*m + j].i1 + l_int_mat[i*k + t].i4*r_int_mat[t*m + j].i3;
				acc[(2*i+1)*m + 2*j+1] += l_int_mat[i*k + t].i3*r_int_mat[t*m + j].i2 + l_int_mat[i*k + t].i4*r_int_mat[t*m + j].i4;
			}
		}
	}

	for (; i < n; i++)
	{
		for (j=0; j < m; j++)
		{
			for (t=0; t<k; t++ ){
				acc[2*i*m + 2*j] += l_int_mat[i*k + t].i1*r_int_mat[t*m + j].i1 + l_int_mat[i*k + t].i2*r_int_mat[t*m + j].i3;
				acc[2*i*m + 2*j+1] += l_int_mat[i*k + t].i1*r_int_mat[t*m + j].i2 + l_int_mat[i*k + t].i2*r_int_mat[t*m + j].i4;
				acc[(2*i+1)*m + 2*j] += l_int_mat[i*k + t].i3*r_int_mat[t*m + j].i1 + l_int_mat[i*k + t].i4*r_int_mat[t*m + j].i3;
				acc[(2*i+1)*m + 2*j+1] += l_int_mat[i*k + t].i3*r_int_mat[t*m + j].i2 + l_int_mat[i*k + t].i4*r_int_mat[t*m + j].i4;
			}
		}
	}

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < m; j++)
		{
			acc[2*i*m + 2*j] +=  - term2[2*j] - term3[2*i] + term4;
			acc[2*i*m + 2*j+1] +=  - term2[2*j + 1] - term3[2*i] + term4;
			acc[(2*i+1)*m + 2*j] += - term2[2*j] - term3[2*i + 1] + term4;
			acc[(2*i+1)*m + 2*j+1] += - term2[2*j + 1] - term3[2*i + 1] + term4;

			result_int_mat[i*m + j].i1 = saturate(round(result_offset.i1 + (l_scale * r_scale/result_scale) * acc[2*i*m + 2*j]));
			result_int_mat[i*m + j].i2 = saturate(round(result_offset.i1 + (l_scale * r_scale/result_scale) * acc[2*i*m + 2*j+1]));
			result_int_mat[i*m + j].i3 = saturate(round(result_offset.i1 + (l_scale * r_scale/result_scale) * acc[(2*i+1)*m + 2*j]));
			result_int_mat[i*m + j].i4 = saturate(round(result_offset.i1 + (l_scale * r_scale/result_scale) * acc[(2*i+1)*m + 2*j+1]));
		}
	}

}



void qmm_trick_AVX(float l_scale, float r_scale, float result_scale, uint4x4_t l_offset, uint4x4_t r_offset, 
	uint4x4_t result_offset, uint4x4_t* l_int_mat, uint4x4_t* r_int_mat, uint4x4_t* result_int_mat, 
	int n, int k, int m){

	int16_t acc11_int[16],acc12_int[16],acc21_int[16],acc22_int[16];
	int16_t acc1,acc2,acc3,acc4;
	uint4x4_t r_elment,l_element;
	// Internally we divide everything by 2 because one uint4x4_t contains 4 integers of 4 bit organize in 2 cols and 2 rows

	n = n/2;
	m = m/2;
	k = k/2;
	int t,i,j;

	// Precompute additional terms
	uint16_t* term2 = (uint16_t*)malloc(sizeof(uint16_t)* m *2);
	uint16_t* term3 = (uint16_t*)malloc(sizeof(uint16_t)* n *2);
	uint16_t term4;
	float scale = l_scale * r_scale/result_scale;
	__m256 scale_avx = _mm256_broadcast_ss(&scale);
	float offset = (float)result_offset.i1;
	__m256 zp_avx = _mm256_broadcast_ss(&offset);
	__m256 trick_m256;
	float qmin=0;
	float qmax=15;
	__m256 qmin_avx = _mm256_broadcast_ss(&qmin);
	__m256 qmax_avx = _mm256_broadcast_ss(&qmax);
	__m256 acc_m256[8];
	uint16_t sum1, sum2;
	__m256i r1,r2;
	__m256i temp[32],temp_t[32];
	__m256i acc11[16],acc12[16],acc21[16],acc22[16],dot_prod1[32],dot_prod2[32];
	float store[8][8];
	// Sum over the entries of each col of rhs and multiply by left offset 
	for(int j =0; j<m; j++){
		sum1 = 0;
		sum2 = 0;
		for(int i = 0; i<k; i++){
			sum1 += r_int_mat[i*m + j].i1 + r_int_mat[i*m + j].i3;
			sum2 += r_int_mat[i*m + j].i2 + r_int_mat[i*m + j].i4;
		}
		term2[2*j] = l_offset.i1 * sum1;
		term2[2*j + 1] = l_offset.i1 * sum2;	
	}

	// Sum over the entries of each row of lhs and multiply by rigt offset
	for (int i=0; i<n; i++){
		sum1 = 0;
		sum2 = 0;
		for (int j=0; j<k; j++){
			sum1 += l_int_mat[i*k + j].i1 + l_int_mat[i*k + j].i2;
			sum2 += l_int_mat[i*k + j].i3 + l_int_mat[i*k + j].i4;
		}
		term3[2*i] = r_offset.i1 * sum1;
		term3[2*i + 1] = r_offset.i1 * sum2;
	}

	term4 = l_offset.i1 * r_offset.i1 * (k * 2);

	for( i=0; i<n; i = i+1){
		for( j=0; j<(m-15); j = j+16){
			
			for (int u = 0; u < 16; u++)
			{
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
				
				for (int u = 0; u < 16; u++)
				{
					uint4x4_to_mm256_row_shuffle(r_int_mat + (t+u)*m + j, &temp[2*u], &temp[2*u+1]);	
				}
				
				transpose(temp,temp_t);

				for (int u = 0; u < 32; u++)
				{
					dot_prod1[u] = _mm256_maddubs_epi16 (r1,temp_t[u]);
					dot_prod2[u] = _mm256_maddubs_epi16 (r2,temp_t[u]);
				}

				for (int u = 0; u < 16; u++)
				{
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

			for (int t = 0; t < 16; t++)
			{
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

			for (int u = 0; i < 8; u+=1)
			{
				acc_m256[u] = _mm256_set_ps(acc22_int[2*u+1], acc21_int[2*u+1], acc12_int[2*u+1], acc11_int[2*u+1], acc22_int[2*u], acc21_int[2*u], acc12_int[2*u],acc11_int[2*u]);
			}
			float trick11 = - term2[2*j] - term3[2*i] + term4;
			float trick12 = - term2[2*j + 1] - term3[2*i] + term4;
			float trick21 = - term2[2*j] - term3[2*i + 1] + term4;
			float trick22 = - term2[2*j + 1] - term3[2*i + 1] + term4;

			trick_m256 = _mm256_set_ps(trick22,trick21,trick12,trick11,trick22,trick21,trick12,trick11);

			for (int u = 0; u < 8; u++)
			{

				acc_m256[u] = _mm256_add_ps(acc_m256[u],trick_m256);
				acc_m256[u] = _mm256_fmadd_ps(scale_avx,acc_m256[u],zp_avx);
				_mm256_store_ps(store[u], _mm256_max_ps( qmin_avx, _mm256_min_ps( qmax_avx, _mm256_round_ps(acc_m256[u],_MM_FROUND_TO_NEAREST_INT))));

				result_int_mat[i*m + j+2*u].i1 = (uint8_t)(store[u][0]);
				result_int_mat[i*m + j+2*u].i2 = (uint8_t)(store[u][1]);
				result_int_mat[i*m + j+2*u].i3 = (uint8_t)(store[u][2]);
				result_int_mat[i*m + j+2*u].i4 = (uint8_t)(store[u][3]);
				result_int_mat[i*m + j+2*u+1].i1 = (uint8_t)(store[u][4]);
				result_int_mat[i*m + j+2*u+1].i2 = (uint8_t)(store[u][5]);
				result_int_mat[i*m + j+2*u+1].i3 = (uint8_t)(store[u][6]);
				result_int_mat[i*m + j+2*u+1].i4 = (uint8_t)(store[u][7]);
				
			}
		}

		for(; j<m; j = j+1){
			acc1 = 0;
			acc2 = 0;
			acc3 = 0;
			acc4 = 0;
			for (int t=0; t<k; t = t+1){
				l_element = l_int_mat[i*k + t]; 
				r_elment = r_int_mat[t*m + j];
				acc1 += (l_element.i1) * (r_elment.i1) + 
						(l_element.i2) * (r_elment.i3);

				acc2 += (l_element.i1) * (r_elment.i2) + 
						(l_element.i2) * (r_elment.i4);

				acc3 += (l_element.i3) * (r_elment.i1) + 
						(l_element.i4) * (r_elment.i3);

				acc4 += (l_element.i3) * (r_elment.i2) + 
						(l_element.i4) * (r_elment.i4);

			}
			
		acc1 = acc1 - term2[2*j] - term3[2*i] + term4;
		acc2 = acc2 - term2[2*j + 1] - term3[2*i] + term4;
		acc3 = acc3 - term2[2*j] - term3[2*i + 1] + term4;
		acc4 = acc4 - term2[2*j + 1] - term3[2*i + 1] + term4;

		result_int_mat[i*m + j].i1 = saturate(round(result_offset.i1 + (l_scale * r_scale/result_scale) * acc1));
		result_int_mat[i*m + j].i2 = saturate(round(result_offset.i1 + (l_scale * r_scale/result_scale) * acc2));
		result_int_mat[i*m + j].i3 = saturate(round(result_offset.i1 + (l_scale * r_scale/result_scale) * acc3));
		result_int_mat[i*m + j].i4 = saturate(round(result_offset.i1 + (l_scale * r_scale/result_scale) * acc4));
		
		}


	}

}
