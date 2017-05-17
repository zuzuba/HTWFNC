#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>

#define qmin 0
#define qmax 15
#define BUFF_SIZE 100000 

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
	*zero_point = saturate(round(-min/(*scale)));
}

void get_min_max(float *d,int rows,int columns, float *mn, float *mx){
	*mx = d[0];
	*mn = *mx;
	float el;
	for(int i = 0; i<rows; i++){
		for(int j = 0; j<columns; j++){
			el = d[i*columns+j];
			*mx = max(*mx,el);
			*mn = min(*mn,el);
		}
	}
}

float * read_csv_mat(const char *filename, int rows, int cols){
	float * mat = (float*)malloc(sizeof(float)* rows * cols);
	char line[BUFF_SIZE];
	char * tok;
	char * ptr;
	int i = 0;
	int j = 0;
	const char * del = ", ";

	FILE* csv = fopen(filename, "r");

	while(fgets(line, BUFF_SIZE, csv)){
		//printf("%s\n", line);
		tok = strtok(line, del);
		j = 0;

		while (tok != NULL){
			mat[i*cols + j] = strtof(tok, &ptr);
			//printf("%s\n", tok);
			tok = strtok(NULL, del);
			j++;
		}
	i++;
	}
	return mat;
}


void print_uint4x4_mat(uint4x4_t* q_mat, int rows, int cols){
	
	for (int i = 0; i<rows/2; i++){
		for (int j = 0; j<cols/2; j++){
			printf("%d  %d  ", q_mat[i*cols/2 + j].i1, q_mat[i*cols/2 + j].i2);
		}
		printf("\n");
		for (int j = 0; j<cols/2; j++){
			printf("%d  %d  ", q_mat[i*cols/2 + j].i3, q_mat[i*cols/2 + j].i4);
		}
		printf("\n");

	}
}


int* get_real_label(float* y_distribution, int data_points, int classes){
	int* label = (int*)malloc(sizeof(int) * data_points);

	for (int i = 0; i<data_points; i++){
		label[i] = 0;
		int running_max = y_distribution[i*classes + 0];
		for (int j = 1; j<classes; j++){
			if (y_distribution[i*classes + j] > running_max){
				running_max = y_distribution[i*classes + j];
				label[i] = j;
			}
		}
	}

	return label;
}


int* get_predicted_label(uint4x4_t* y_distribution, int data_points, int classes){
		int* label = (int*)malloc(sizeof(int) * data_points);

		data_points = data_points/2;
		classes = classes/2;

		for (int i=0; i<data_points; i++){
			
			label[i*2] = -1;
			label[i*2 + 1] = -1;
			int running_max1 = INT_MIN;
			int running_max2 = INT_MIN;

			for (int j=0; j<classes; j++){
				if (y_distribution[i*classes + j].i1 >running_max1){
					running_max1 = y_distribution[i*classes + j].i1;
					label[i*2] = j * 2;
				}


				if (y_distribution[i*classes + j].i2 >running_max1){
					running_max1 = y_distribution[i*classes + j].i2;
					label[i*2] = j * 2 + 1;
				}

				if (y_distribution[i*classes + j].i3 >running_max2){
					running_max2 = y_distribution[i*classes + j].i3;
					label[i*2 + 1] = j * 2;
				}

				if (y_distribution[i*classes + j].i4 >running_max2){
					running_max2 = y_distribution[i*classes + j].i4;
					label[i*2 + 1] = j * 2 + 1;
				}

			}
		}

		return label;
}


