#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>

float inline max(float a, float b);
float inline min(float a, float b);
int saturate(float a);

void quantize_parameter(float min, float max, float *scale, float *zero_point);
void get_min_max(float *d,int rows,int columns, float *min, float *max);

typedef struct unsigned_4_bit {
   uint8_t i : 4;
} uint4x1_t;

#endif
