#include "utils.h"
#include "naive_qmm.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define FEATURES 784
#define CLASSES 10
#define TEST_POINTS 1000


int main(){
	// Reading a test matrix
	float *W_test = read_csv_mat("weight_test.csv", 3, 2); // Buffer size for row is defined in utils and might be too small for big matrices

	for (int i=0; i<3; i++){
		for (int j=0; j<2; j++){
			printf("%f\t", W_test[i*2 + j]);
		}
		printf("\n");
	}

	float *W = read_csv_mat("weight.csv", FEATURES, CLASSES);
	float *b = read_csv_mat("bias.csv", CLASSES, 1);
	float *x_test = read_csv_mat("x_test.csv", TEST_POINTS, FEATURES);
	float *y_test = read_csv_mat("y_test.csv", TEST_POINTS, CLASSES);
	
	//...quantize W, b, new inputs x_test...
	//...quantized W*x_test+b...
	//...quantized softmax...



	return 0;
}