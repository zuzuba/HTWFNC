#include <stdio.h>
#include <stdint.h>
#include "utils.h"
#include "trick_vector.h"
#include <math.h>
#include <stdlib.h>
#include <immintrin.h> 


void trick_vector_naive(uint4x4_t* l_int_mat, uint4x4_t* r_int_mat,uint4x4_t l_offset, uint4x4_t r_offset,int n, int k, int m,
	uint16_t* term2, uint16_t* term3, uint16_t* term4){

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

*term4 = l_offset.i1*r_offset.i1*(k * 2);

}