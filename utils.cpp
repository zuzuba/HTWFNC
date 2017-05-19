#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <immintrin.h> 

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
			printf("%d %d ", q_mat[i*cols/2 + j].i1, q_mat[i*cols/2 + j].i2);
		}
		printf("\n");
		for (int j = 0; j<cols/2; j++){
			printf("%d %d ", q_mat[i*cols/2 + j].i3, q_mat[i*cols/2 + j].i4);
		}
		printf("\n");

	}
}

void print_uint8_mm256i(__m256i a){
		printf("%d ", _mm256_extract_epi8(a,0)); printf("%d ", _mm256_extract_epi8(a,1));
		printf("%d ", _mm256_extract_epi8(a,2)); printf("%d ", _mm256_extract_epi8(a,3));
		printf("%d ", _mm256_extract_epi8(a,4)); printf("%d ", _mm256_extract_epi8(a,5));
		printf("%d ", _mm256_extract_epi8(a,6)); printf("%d ", _mm256_extract_epi8(a,7)); 
		printf("%d ", _mm256_extract_epi8(a,8)); printf("%d ", _mm256_extract_epi8(a,9)); 
		printf("%d ", _mm256_extract_epi8(a,10)); printf("%d ", _mm256_extract_epi8(a,11)); 
		printf("%d ", _mm256_extract_epi8(a,12)); printf("%d ", _mm256_extract_epi8(a,13)); 
		printf("%d ", _mm256_extract_epi8(a,14)); printf("%d ", _mm256_extract_epi8(a,15)); 
		printf("%d ", _mm256_extract_epi8(a,16)); printf("%d ", _mm256_extract_epi8(a,17)); 
		printf("%d ", _mm256_extract_epi8(a,18)); printf("%d ", _mm256_extract_epi8(a,19)); 
		printf("%d ", _mm256_extract_epi8(a,20)); printf("%d ", _mm256_extract_epi8(a,21)); 
		printf("%d ", _mm256_extract_epi8(a,22)); printf("%d ", _mm256_extract_epi8(a,23)); 
		printf("%d ", _mm256_extract_epi8(a,24)); printf("%d ", _mm256_extract_epi8(a,25));
		printf("%d ", _mm256_extract_epi8(a,26)); printf("%d ", _mm256_extract_epi8(a,27)); 
		printf("%d ", _mm256_extract_epi8(a,28)); printf("%d ", _mm256_extract_epi8(a,29)); 
		printf("%d ", _mm256_extract_epi8(a,30)); printf("%d ", _mm256_extract_epi8(a,31)); 
		printf("\n");
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

void uint4x4_to_mm256_row(uint4x4_t* a, __m256i *b1, __m256i *b2){
	*b1 = _mm256_set_epi8(a[15].i2, a[15].i1, a[14].i2, a[14].i1, a[13].i2, a[13].i1, a[12].i2, a[12].i1, 
		a[11].i2, a[11].i1, a[10].i2, a[10].i1, a[9].i2, a[9].i1, a[8].i2, a[8].i1, a[7].i2, a[7].i1, 
		a[6].i2, a[6].i1, a[5].i2, a[5].i1, a[4].i2, a[4].i1, a[3].i2, a[3].i1, a[2].i2, a[2].i1, a[1].i2,
		 a[1].i1, a[0].i2, a[0].i1);

	*b2 = _mm256_set_epi8(a[15].i4, a[15].i3, a[14].i4, a[14].i3, a[13].i4, a[13].i3, a[12].i4, a[12].i3, 
		a[11].i4, a[11].i3, a[10].i4, a[10].i3, a[9].i4, a[9].i3, a[8].i4, a[8].i3, a[7].i4, a[7].i3, 
		a[6].i4, a[6].i3, a[5].i4, a[5].i3, a[4].i4, a[4].i3, a[3].i4, a[3].i3, a[2].i4, a[2].i3, a[1].i4,
		 a[1].i3, a[0].i4, a[0].i3);
}

void uint4x4_to_mm256_column(uint4x4_t* a, __m256i *b1, __m256i *b2){
	*b1 = _mm256_set_epi8(a[15].i3, a[15].i1, a[14].i3, a[14].i1, a[13].i3, a[13].i1, a[12].i3, a[12].i1, 
		a[11].i3, a[11].i1, a[10].i3, a[10].i1, a[9].i3, a[9].i1, a[8].i3, a[8].i1, a[7].i3, a[7].i1, 
		a[6].i3, a[6].i1, a[5].i3, a[5].i1, a[4].i3, a[4].i1, a[3].i3, a[3].i1, a[2].i3, a[2].i1, a[1].i3,
		a[1].i1, a[0].i3, a[0].i1);

	*b2 = _mm256_set_epi8(a[15].i4, a[15].i2, a[14].i4, a[14].i2, a[13].i4, a[13].i2, a[12].i4, a[12].i2, 
		a[11].i4, a[11].i2, a[10].i4, a[10].i2, a[9].i4, a[9].i2, a[8].i4, a[8].i2, a[7].i4, a[7].i2, 
		a[6].i4, a[6].i2, a[5].i4, a[5].i2, a[4].i4, a[4].i2, a[3].i4, a[3].i2, a[2].i4, a[2].i2, a[1].i4,
		a[1].i2, a[0].i4, a[0].i2);
}

uint16_t dot_prod_AVX(__m256i a, __m256i b){
	__m256i c,t1,t2,t3,t4,t5,t6; 
	c = _mm256_maddubs_epi16 (a,b);
	uint16_t result =0;
	t1= _mm256_permute4x64_epi64(c,177);
	t2= _mm256_permute4x64_epi64(c,142);
	t3= _mm256_permute4x64_epi64(c,27);
	t4 = _mm256_add_epi16(c,t1);
	t5 = _mm256_add_epi16(t2,t3);
	t6 = _mm256_add_epi16(t4,t5);

	return _mm256_extract_epi16(t6,0)+_mm256_extract_epi16(t6,1)+_mm256_extract_epi16(t6,2)+_mm256_extract_epi16(t6,3);
}

