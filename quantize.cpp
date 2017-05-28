#include<iostream>
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

	float t1, t2, t3, t4;
	for(int i = 0; i<rows; i = i+2){
		for(int j = 0; j<columns; j = j+2){
			t1 = zero_point + d[i*columns + j]/scale;
			t2 = zero_point + d[i*columns + j + 1]/scale;
			t3 = zero_point + d[(i + 1)*columns + j]/scale;
			t4 = zero_point + d[(i + 1)*columns + j + 1]/scale;
			q[i/2*columns/2 + j/2].i1 = (uint8_t)saturate(round(t1));
			q[i/2*columns/2 + j/2].i2 = (uint8_t)saturate(round(t2));
			q[i/2*columns/2 + j/2].i3 = (uint8_t)saturate(round(t3));
			q[i/2*columns/2 + j/2].i4 = (uint8_t)saturate(round(t4));
		}
	}	
}

void quantize_AVX(float *d, uint4x4_t *q, float *min, float *max, int rows, int columns){

	float scale, zero_point;
	get_min_max_AVX(d,rows,columns,min,max);
	quantize_parameter(*min,*max,&scale,&zero_point);
	scale = 1/scale;
	float qmin = 0.0;
	float qmax = 15.0;
	int j;
	__m256 scale_avx = _mm256_broadcast_ss(&scale);
	__m256 zp_avx = _mm256_broadcast_ss(&zero_point);
	__m256 qmin_avx = _mm256_broadcast_ss(&qmin);
	__m256 qmax_avx = _mm256_broadcast_ss(&qmax);
	__m256 upper_row;
	__m256 lower_row;

//BEGIN_WIP
	/*This part is currently work in progress. Since the struct seems to be reading bits in reverse order, 
	 * we should be able to remedy the situation by simply reversing the order in which we shift elements.
	 * Currenlty, if we have two rows of the matrix, we place 8 elements of each row in a register, upper_row and lower_row
	 * Then, we perform quantization and then staturate and round. Subsequently, we shift the least-significant 4-bits
	 * by top_shift (for the upper_row) and bottom_shift (for the lower_row)
	 * Using the top_shift and bottom_shift below should reverse the order of the members in the struct when we write straight
	 * to the memory location of the struct, but it does not.
	 * Very Frustrating.
	 */
	//__m256 top_shift = _mm256_set_ps(16.0,1.0,16.0,1.0,16.0,1.0,16.0,1.0);
	//__m256 bottom_shift = _mm256_set_ps(4096.0,256.0,4096.0,256.0,4096.0,256.0,4096.0,256.0);
//END_WIP
	
	__m256 top_shift = _mm256_set_ps(256.0,4096.0,256.0,4096.0,256.0,4096.0,256.0,4096.0);
	__m256 bottom_shift = _mm256_set_ps(1.0,16.0,1.0,16.0,1.0,16.0,1.0,16.0);

	uint32_t temp_upper[8];
	uint16_t fi[16];
	uint32_t temp_lower[8];
	uint16_t gi[16];
	float t1, t2, t3, t4;
	uint16_t temp;
	for(int i = 0; i<rows; i = i+2){
		for(j = 0; j<(columns-7); j = j+8){

			/* Ok, so let's take the following example. Since we are using Floats, there are 8 floats in every AVX register.
			 * We use 2 registers: upper_row and lower_row. There are 8 floats in upper_row and 8 in lower_row.
			 * So, our registers look like this
			 * upper_row = [ ur1, ur2, ur3, ur4, ur5, ur6, ur7 ]
			 * lower_row = [ lr1, lr2, lr3, lr4, lr5, lr6, lr7 ]
			 */

			upper_row = _mm256_loadu_ps(d+(i*columns+j));
			lower_row = _mm256_loadu_ps(d+((i+1)*columns+j));
			
			/* Now, we apply FMADD to get the desired result x/scale + zero_point
			 * As you can see, scale_avx uses the reciprocal of the scale float, so we
			 * can use FMADD to perform this operation as below 
			 */
			
			upper_row = _mm256_fmadd_ps(scale_avx,upper_row,zp_avx);
			lower_row = _mm256_fmadd_ps(scale_avx,lower_row,zp_avx);

			/* Now, we do rounding and saturation in AVX using intrinsics, note that
			 * qmin_avx and qmax_avx are float AVX registers
			 */ 
			
			upper_row = _mm256_max_ps( qmin_avx, _mm256_min_ps( qmax_avx, _mm256_round_ps(upper_row,_MM_FROUND_TO_NEAREST_INT)));
        		lower_row = _mm256_max_ps( qmin_avx, _mm256_min_ps( qmax_avx, _mm256_round_ps(lower_row,_MM_FROUND_TO_NEAREST_INT)));
        		
			/* Now the upper_row and lower_row have fthe float values that we need. They are rounded and saturated
			 * All that remains is to cast them to uint4_t
			 * In order to achieve the cast, we begin by casting each float to a 32-bit uint, but since we know
			 * that for each 32-bit uint, only the first 4 bits are non-zero (due to our saturation above), we
			 * want to shift them first. Every element of even index in the upper_row AVX register has to be shifted 12 bits to
			 * to the left and every element of uneven index has to be shifted 8-bits to the left.
			 * For lower_row, every element of even index has to be shifted 4-bits to the left, e.g.
			 * upper_row = [ ur1<<12, ur2<<8, ur3<<12, ur4<<8, ... ]
			 * lower_row = [ lr1<<4, lr2, lr3<<4, lr4, ... ]
			 * Since shifting different number of bits for different indexes is somewhat cumbersome, a simple floating point
			 * multiplication does the job just fine, so I allocated top_shift and bottom_shift to do the job.
			 * Once done, we just cast to 32-bit uints.
			 */

			upper_row = _mm256_cvttps_epi32( _mm256_mul_ps( top_shift, upper_row) );
        		lower_row = _mm256_cvttps_epi32( _mm256_mul_ps( bottom_shift, lower_row) );

			/* Now we or the two registers upper_row and lower_row
			 * The result is that i1 and i3 are in the elements of even index, while i2 and i4 are in the elements of 
			 * uneven index. That is:
			 * lower_row = [ (ur1<<12) | (lr1<<4) , (ur2<<8) | (lr2) , (ur3<<12) | (lr3<<4) , (ur4<<8) | (lr4) , ... ]
			 */

			lower_row = _mm256_or_si256(upper_row, lower_row);

			/* Now we need to OR the lower_row register with itself (shifted left 32-bits).
			 * BUT, we don't actually need the entire register to be shifted left, we only need every other element to be shifted 32-bits
			 * to the left of where it is in lower_row (because we only need to OR pairwise neighoring 32-bit elements). Coincidentally, we can achieve this
			 * by actually cycling the elements in a lane, i.e. 0,1,2,3 -> 1,2,3,0 for both lanes. Therefore, we can just use the permute functionality
			 * that places the 32-bit ints at index 1,3,5,7 in the positions at index 0,2,4,6
			 * NOTE:
			 * lower_row = [ (ur1<<12) | (lr1<<4) , (ur2<<8) | (lr2) , (ur3<<12) | (lr3<<4) , (ur4<<8) | (lr4) , ... ]
			 * _mm256_permute_ps( lower_row, 57 ) =
			 * 		[ (ur2<<8) | (lr2) , (ur3<<12) | (lr3<<4) , (ur4<<8) | (lr4) , (ur1<<12) | (lr1<<4) , ... ] (analogous for the other lane)
			 */

			upper_row = _mm256_or_si256(lower_row, _mm256_permute_ps(lower_row, 57 ) );

			/* Since we only
			 * 1. Require the 16 most significant bits of every 32-bit int
			 * 2. Only require every other 32-bit register in the first place
			 * We store the entire AVX register here in the array: uint16_t fi[16]
			 * We will only need fi[0], fi[4], fi[8], fi[12]
			 */

			_mm256_store_si256( (__m256i*)&fi, upper_row);

		//BEGIN_DEBUGGING

			/* For debugging purposes, we can keep lower_row, which is "almost" perfect
			 * If you use gi instead of fi, you need to pairwise OR together neigboring 32-bit registers (more precisely OR together every other uint16_t
			 * corresponding to neigboring slots in the AVX register
			 */

			//_mm256_store_si256( (__m256i*)&gi, lower_row);
			 
			/* IF USING gi: Now, recall that we were working with 32-bit floats, but with saturation and rounding, we ended up with 32-bit ints (of which only the first
			 * 4-bits could every be set. The array fi is an array of uint16_t's, so therefore every other element (the elements with uneven index) in fi is
			 * always 0. Therefore, fi[0] and fi[2] are in fact the two pieces that we need to compute the first struct.
			 * We OR fi[0] and fi[2] in temp to arrive that the desired values.
			 */

			//temp = (gi[0] | gi[2]);
			
		//END_DEBUGGING

			/* IF USING fi: Nothing to be done. Every 4th uint16_t is the exact 16-bits needed to encode the 4 4-bit integers
			 */

			temp = fi[0];
			
		//BEGIN_WIP

			/* TODO: This, doesn't work! Even though it should */
			//UNCOMMENT HERE IF YOU WOULD LIKE TO IMPROVE THE CODE
			//*(uint16_t*)(q+(i/2*columns/2 + j/2)) = temp;

		//END_WIP

			/* This does work! Therefore, you see that the values in temp are correct.
			 * Shifting temp by 12 bits, i.e. temp>>12, gives us the 4 most siginificant bits
			 * Shifting temp by 8 bits, i.e. temp>>8, gives us the 8 most significant bits
			 * Shifting temp by 4 bits, i.e. temp>>4, gives us the 12 most significant bits
			 * Not shifting, gives us all 16 bits
			 * Since the members of our struct are uint8_t X : 4, only the 4 least-significant bits
			 * of the r-value are ever taken, so there is not need to cast here.
			 */

			q[i/2*columns/2 + j/2].i1 = temp>>12;
			q[i/2*columns/2 + j/2].i2 = temp>>8;
			q[i/2*columns/2 + j/2].i3 = temp>>4;
			q[i/2*columns/2 + j/2].i4 = temp;
			
		//BEGIN_DEBUGGING
			/*
			bool b;
			uint16_t *t = (uint16_t*)(q+(i/2*columns/2+j/2));
			std::cout << "BITS: ";
			for (int k=15; k>=0; k--){
				b = (*t & (1<<k)) > 0 ? 1 : 0;
				std::cout << b;	
			}
			std::cout << temp << " " << fi[0] << std::endl;
			std::cout << std::endl << "DECIMALS: " << (temp>>12 & 15) << " " << (temp>>8 & 15) << " " << (temp>>4 & 15) << " " << (temp & 15);
			std::cout << std::endl << "Member Values: " << (int) q[i/2*columns/2+j/2].i1 << " " << (int) q[i/2*columns/2 + j/2].i2 << " " << (int) q[i/2*columns/2 + j/2].i3 << " " << (int) q[i/2*columns/2 + j/2].i4  << std::endl << std::endl;
			*/
		//END_DEBUGGING

			

			//temp = (gi[4] | gi[6]); //IF USING DEBUGGING gi as oppose to complete fi
			temp = fi[4];
			q[i/2*columns/2 + j/2 + 1].i1 = temp>>12;
			q[i/2*columns/2 + j/2 + 1].i2 = temp>>8;
			q[i/2*columns/2 + j/2 + 1].i3 = temp>>4;
			q[i/2*columns/2 + j/2 + 1].i4 = temp;
			//temp = (gi[8] | gi[10]); //IF USING DEBUGGING gi as oppose to complete fi
			temp = fi[8];
			q[i/2*columns/2 + j/2 + 2].i1 = temp>>12;
			q[i/2*columns/2 + j/2 + 2].i2 = temp>>8;
			q[i/2*columns/2 + j/2 + 2].i3 = temp>>4;
			q[i/2*columns/2 + j/2 + 2].i4 = temp;
			//temp = (gi[12] | gi[14]); //IF USING DEBUGGING gi as oppose to complete fi
			temp = fi[12];
			q[i/2*columns/2 + j/2 + 3].i1 = temp>>12;
			q[i/2*columns/2 + j/2 + 3].i2 = temp>>8;
			q[i/2*columns/2 + j/2 + 3].i3 = temp>>4;
			q[i/2*columns/2 + j/2 + 3].i4 = temp;
		}	
		for(; j<columns; j = j+2){
			t1 = zero_point + d[i*columns + j]*scale;
			t2 = zero_point + d[i*columns + j + 1]*scale;
			t3 = zero_point + d[(i + 1)*columns + j]*scale;
			t4 = zero_point + d[(i + 1)*columns + j + 1]*scale;
			q[i/2*columns/2 + j/2].i1 = (uint8_t)saturate(round(t1));
			q[i/2*columns/2 + j/2].i2 = (uint8_t)saturate(round(t2));
			q[i/2*columns/2 + j/2].i3 = (uint8_t)saturate(round(t3));
			q[i/2*columns/2 + j/2].i4 = (uint8_t)saturate(round(t4));
		}
	}	

}


void dequantize(double **d, uint8_t **q, double *mn, double *mx, int n){
	double delta = (*mx - *mn)/256; // size of linear cell
	double runner = 0.0;
	for(int i = 0; i<n; i++){
		for(int j = 0; j<n; j++){
			d[i][j] = *mn + q[i][j]*delta;
		}
	}
}
