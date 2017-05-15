#include "utils.h"
#include "naive_qmm.h"
#include "naive_quantize.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define FEATURES 786
#define CLASSES 10
#define TEST_POINTS 1000


int main(){
	// Reading a test matrix
	// float *W_test = read_csv_mat("data/weight_test.csv", 3, 2); // Buffer size for row is defined in utils and might be too small for big matrices

	// for (int i=0; i<3; i++){
	// 	for (int j=0; j<2; j++){
	// 		printf("%f\t", W_test[i*2 + j]);
	// 	}
	// 	printf("\n");
	// }

	float *W = read_csv_mat("data/weight.csv", FEATURES, CLASSES);
	float *x_test = read_csv_mat("data/x_test.csv", TEST_POINTS, FEATURES);
	float *y_test = read_csv_mat("data/y_test.csv", TEST_POINTS, CLASSES);
	
	uint4x4_t *W_q = (uint4x4_t*)malloc(sizeof(uint4x4_t) * FEATURES/2 * CLASSES/2);
	uint4x4_t *x_q = (uint4x4_t*)malloc(sizeof(uint4x4_t) * TEST_POINTS/2 * FEATURES/2);
	uint4x4_t *result_q = (uint4x4_t*)malloc(sizeof(uint4x4_t) * TEST_POINTS/2 * CLASSES/2);
	float min_w, max_w, scale_w, zero_point_w;
	float min_x, max_x, scale_x, zero_point_x;
	float scale_result, zero_point_result;

	quantize_4x4(W, W_q, &min_w, &max_w, FEATURES, CLASSES);
	quantize_4x4(x_test, x_q, &min_x, &max_x, TEST_POINTS, CLASSES);

	quantize_parameter(min_w, max_w, &scale_w, &zero_point_w);
	quantize_parameter(min_x, max_x, &scale_x, &zero_point_x);

	uint4x4_t offset_w, offset_x, offset_result;
	offset_w.i1 = zero_point_w;
	offset_x.i1 = zero_point_x;

	scale_result = 2.53;
	offset_result.i1 = 7;

	// Notice we need to do x * W and not the other way around
	qmm_naive(scale_x, scale_w, scale_result, offset_x,  offset_w, 
	offset_result, W_q, x_q, result_q, TEST_POINTS, 
	FEATURES, CLASSES);	

	
	//Prediciton
	int real_ind_1, real_ind_2;
	int correct_predictions = 0;
	for (int i=0; i<TEST_POINTS; i = i+2){
		int running_ind1 = 0;
		int running_ind2 = 0;
		int runnin_max1 = 0;
		int runnin_max2 = 0;
		for (int j=0; j<CLASSES; j = j+2){

			if (y_test[i*CLASSES + j] == 1 || y_test[i*CLASSES + j +1] == 1){
				real_ind_1 = j;
			}

			if (y_test[(i +1)*CLASSES] == 1 || y_test[(i+1)*CLASSES + j +1] == 1){
				real_ind_2 = j;
			}
			if(result_q[i/2*CLASSES/2 + j/2].i1 > runnin_max1){
				runnin_max1 = result_q[i/2*CLASSES/2 + j/2].i1;
				running_ind1 = j/2;
			}
			if(result_q[i/2*CLASSES/2 + j/2].i2 > runnin_max1){
				runnin_max1 = result_q[i/2*CLASSES/2 + j/2].i2;
				running_ind1 = j/2 + 1;
			}

			if(result_q[i/2*CLASSES/2 + j/2].i3 > runnin_max2){
				runnin_max2 = result_q[i/2*CLASSES/2 + j/2].i3;
				running_ind2 = j/2;
			}
			if(result_q[i/2*CLASSES/2 + j/2].i4 > runnin_max2){
				runnin_max2 = result_q[i/2*CLASSES/2 + j/2].i4;
				running_ind2 = j/2 + 1;
			}


		}
		//printf("%d\t%d\n", running_ind1, running_ind2);
		if (real_ind_1 ==running_ind1) correct_predictions++;
		if (real_ind_2 ==running_ind2) correct_predictions++;
			
	}
	//...quantized softmax...

	printf("Ciao\n");
	printf("Correct predictions:\t %d\n", correct_predictions);

	return 0;
}