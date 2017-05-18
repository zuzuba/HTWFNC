#include "utils.h"
#include "qmm.h"
#include "quantize.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define FEATURES 786
#define CLASSES 10
#define TEST_POINTS 10000


int main(){
	 
	// Read the trained network and the test points
	float *W = read_csv_mat("data/weight.csv", FEATURES, CLASSES);
	float *x_test = read_csv_mat("data/x_test.csv", TEST_POINTS, FEATURES);
	float *y_test = read_csv_mat("data/y_test.csv", TEST_POINTS, CLASSES);
	
	// Initialize quantized variables for the network
	uint4x4_t *W_q = (uint4x4_t*)malloc(sizeof(uint4x4_t) * FEATURES/2 * CLASSES/2);
	uint4x4_t *x_q = (uint4x4_t*)malloc(sizeof(uint4x4_t) * TEST_POINTS/2 * FEATURES/2);
	uint4x4_t *result_q = (uint4x4_t*)malloc(sizeof(uint4x4_t) * TEST_POINTS/2 * CLASSES/2);
	float min_w, max_w, scale_w, zero_point_w;
	float min_x, max_x, scale_x, zero_point_x;
	float scale_result, zero_point_result;

	// Quantization step
	quantize_4x4(W, W_q, &min_w, &max_w, FEATURES, CLASSES);
	quantize_4x4(x_test, x_q, &min_x, &max_x, TEST_POINTS, FEATURES);

	quantize_parameter(min_w, max_w, &scale_w, &zero_point_w);
	quantize_parameter(min_x, max_x, &scale_x, &zero_point_x);

	uint4x4_t offset_w, offset_x, offset_result;
	offset_w.i1 = zero_point_w;
	offset_x.i1 = zero_point_x;

	scale_result = 2.53;
	offset_result.i1 = 7;

	// Notice we need to do x * W and not the other way around
	qmm_trick_AVX(scale_x, scale_w, scale_result, offset_x,  offset_w, 
	offset_result, x_q, W_q, result_q, TEST_POINTS, 
	FEATURES, CLASSES);	

	// Prediction
	printf("\n-------------------Starting prediction--------------------\n");
	int *labels = get_real_label(y_test, TEST_POINTS, CLASSES);
	int *labels_predicted = get_predicted_label(result_q, TEST_POINTS, CLASSES);
	int correct_predictions_new = 0;
	for (int i=0; i<TEST_POINTS; i++){
		if (labels[i] == labels_predicted[i]) correct_predictions_new++;
	}
	printf("\nCorrect predictions:\t %f%%\n", (float)correct_predictions_new/TEST_POINTS*100);


	return 0;
}