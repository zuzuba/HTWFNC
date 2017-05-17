#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>

float inline max(float a, float b);
float inline min(float a, float b);
int saturate(float a);

void quantize_parameter(float min, float max, float *scale, float *zero_point);
void get_min_max(float *d,int rows,int columns, float *min, float *max);

float * read_csv_mat(const char *filename, int rows, int cols);



typedef struct unsigned_4_bit {
   uint8_t i : 4;
} uint4x1_t;


/*
Internal structure of uint4x4_t data type considering it represents 4 elements in a matrix
-----------------
| 	i1	|	i2	|
-----------------
| 	i3	|	i4	|
-----------------
*/
typedef struct unsigned_4x4_bit
{
	uint8_t i1 : 4;
	uint8_t i2 : 4;
	uint8_t i3 : 4;
	uint8_t i4 : 4;
}uint4x4_t;

void print_uint4x4_mat(uint4x4_t* q_mat, int rows, int cols);

int* get_real_label(float* y_distribution, int data_points, int classes);
int* get_predicted_label(uint4x4_t* y_distribution, int data_points, int classes);


#endif
