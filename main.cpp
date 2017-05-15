#include "utils.h"
#include "naive_qmm.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define ROWS 784
#define COLS 10


int main(){
	// Reading a test matrix
	float *W_test = read_csv_mat("weight_test.csv", 3, 2); // Buffer size for row is defined in utils and might be too small for big matrices

	for (int i=0; i<3; i++){
		for (int j=0; j<2; j++){
			printf("%f\t", W_test[i*2 + j]);
		}
		printf("\n");
	}

	float *W = read_csv_mat("weight.csv", ROWS, COLS);
	float *b = read_csv_mat("bias.csv", ROWS, 1);
	//...quantize W, b, new inputs x_test...
	//...quantized W*x_test+b...
	//...quantized softmax...



	return 0;
}