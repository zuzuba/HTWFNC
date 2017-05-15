#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define qmin 0
#define qmax 15
#define BUFF_SIZE 1024 

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
