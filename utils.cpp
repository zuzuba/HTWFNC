#include "utils.h"
#include <stdio.h>

#define qmin 0
#define qmax 15

// struct unsigned_4_bit{
//    uint8_t i : 4;
// };


float inline max(float a, float b){
	return a < b ? b : a;
}

float inline min(float a, float b){
	return a < b ? a : b;
}

int saturate(float a){
	return (int)max(qmin,min(qmax,a));
}

void quantize_parameter(float min, float max, float *scale, float *zero_point){
	*scale = (max - min)/(qmax-qmin);
	*zero_point = -min/(*scale);
}

void get_min_max(float *d,int rows,int columns, float *mn, float *mx){
	*mx = d[0];
	*mn = *mx;
	float el;
	for(int i = 0; i<rows; i++){
		for(int j = 0; j<columns; j++){
			el = d[i*rows+columns];
			*mx = max(*mx,el);
			*mn = min(*mn,el);
		}
	}
}
