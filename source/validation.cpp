#include "utils.h"
#include "quantize.h"
#include "qmm.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#define TEST_FAILED -1
#define FEATURES 786
#define CLASSES 10
#define TEST_POINTS 10000
#define IS_USE_MNIST_DATA false


/* prototype of the function you need to optimize */
typedef void(*quantize_pointer)(float *, uint4x4_t*,  float *, float *, int, int);
typedef void(*qmm_pointer_4x4)(float, float, float, uint4x4_t, uint4x4_t, uint4x4_t, 
	uint4x4_t*, uint4x4_t*, uint4x4_t*, int , int, int);

#define MAX_FUNCS 32

void register_functions();
void add_function(quantize_pointer f, char *name, int flop);
int validation_quant(quantize_pointer f, float *d, uint4x4_t *q_answer);

void register_functions_4x4();
void add_function_4x4(qmm_pointer_4x4 f, char *name, int flop);
int validation_qmm4x4(qmm_pointer_4x4 f,float l_scale, float r_scale, float result_scale, uint4x4_t l_offset, uint4x4_t r_offset, 
	uint4x4_t result_offset, uint4x4_t* l_int_mat, uint4x4_t* r_int_mat, uint4x4_t* result_array, 
	int n, int k, int m);

/* Global vars, used to keep track of student functions */
quantize_pointer userFuncs[MAX_FUNCS];
char *funcNames[MAX_FUNCS];
int funcFlops[MAX_FUNCS];
int numFuncs = 0;

qmm_pointer_4x4 userFuncs_4x4[MAX_FUNCS];
char *funcNames_4x4[MAX_FUNCS];
int funcFlops_4x4[MAX_FUNCS];
int numFuncs_4x4 = 0;


void register_functions()
{	
	add_function(&quantize_4x4, (char *)"naive quantize",7);
	// Add your functions here
	// add_function(&your_function, "function: Optimization X", nrflops);
}
void register_functions_4x4()
{	
	add_function_4x4(&qmm_naive, (char *)"naive qmm",7);
    add_function_4x4(&qmm_trick, (char *)"trick qmm",7);
	// Add your functions here
	// add_function(&your_function, "function: Optimization X", nrflops);
	
}

void add_function(quantize_pointer f, char *name, int flops)
{	
	if (numFuncs >= MAX_FUNCS)
	{
		printf("Couldn't register %s, too many functions registered (Max: %d)",
			name, MAX_FUNCS);
		return;
	}

	userFuncs[numFuncs] = f;
	funcNames[numFuncs] = name;
	funcFlops[numFuncs] = flops;

	numFuncs++;
}
void add_function_4x4(qmm_pointer_4x4 f, char *name, int flops)
{	
	if (numFuncs_4x4 >= MAX_FUNCS)
	{
		printf("Couldn't register %s, too many functions registered (Max: %d)",
			name, MAX_FUNCS);
		return;
	}

	userFuncs_4x4[numFuncs_4x4] = f;
	funcNames_4x4[numFuncs_4x4] = name;
	funcFlops_4x4[numFuncs_4x4] = flops;

	numFuncs_4x4++;
}

int main() {
    // Read the trained network and the test points

	float *W;
	float *x_test;
	float *y_test;
	if(IS_USE_MNIST_DATA) {
		W = read_csv_mat("data/weight.csv", FEATURES, CLASSES);
		x_test = read_csv_mat("data/x_test.csv", TEST_POINTS, FEATURES);
		y_test = read_csv_mat("data/y_test.csv", TEST_POINTS, CLASSES);
	}else {
		W = generate_rand_mat(FEATURES, CLASSES);
		x_test = generate_rand_mat(TEST_POINTS, FEATURES);
		y_test = generate_rand_mat(TEST_POINTS, CLASSES);
	}

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
	qmm_naive(scale_x, scale_w, scale_result, offset_x,  offset_w, 
	offset_result, x_q, W_q, result_q, TEST_POINTS, 
	FEATURES, CLASSES);	
	
	//////////////////////////////////////////////////////////////////
    // We run validation here, compare the output with the vanila version output
    printf("------Testing quantization-----\n");
	int verbosity = 2;
	int test_success = 0;
    
    // Initialize the vectors of functions, function names and function flops
	register_functions();
    // Message if there are zero functions
	if (numFuncs == 0)
	{
		printf("No functions registered - nothing for driver to do\n");
		printf("Register functions by calling register_func(f, name)\n");
		printf("in register_funcs()\n");

		return TEST_FAILED;
	}
	printf("\n%d quantization functions registered\n", numFuncs);

	// Test of vanilla quantization implementation first
	test_success = validation_quant(userFuncs[0], W, W_q);
	if (test_success == 0)
	{
        printf("Vanilla implementation failed test!!\n");
		return TEST_FAILED;
	}
    for(int a = 1; a < numFuncs; a ++) {
        test_success = validation_quant(userFuncs[a], W, W_q);
        if (test_success == 0)
        {
            printf(" %d th quantization implementation %s failed test!!\n", a, funcNames[a]);
            return TEST_FAILED;
        }
    }
    
	 // 4x4 section
    // Initialize the vectors of functions, function names and function flops
	register_functions_4x4();
	// Message if there are zero functions
	if (numFuncs_4x4 == 0)
	{
		printf("No 4x4 functions registered - nothing for driver to do\n");
		printf("Register functions by calling register_func(f, name)\n");
		printf("in register_funcs_4x4()\n");

		return TEST_FAILED;
	}
	printf("\n%d 4x4 functions registered\n", numFuncs_4x4);
    // Test of vanilla qmm implementation first
	test_success = validation_qmm4x4(userFuncs_4x4[0],scale_x, scale_w, scale_result, offset_x,  offset_w, 
	offset_result, x_q, W_q, result_q, TEST_POINTS, 
	FEATURES, CLASSES);
	if (test_success == 0)
	{
		printf("Vanilla 4x4 qmm implementation failed test!!\n");
		return TEST_FAILED;
	}
    for(int a = 1; a < numFuncs_4x4; a ++) {
        test_success = validation_qmm4x4(userFuncs_4x4[a],scale_x, scale_w, scale_result, offset_x,  offset_w, 
                        offset_result, x_q, W_q, result_q, TEST_POINTS, 
                        FEATURES, CLASSES);
        if (test_success == 0)
        {
            printf(" %d th qmm implementation %s failed test!!\n", a, funcNames_4x4[a]);
            return TEST_FAILED;
        }
    }



}

int validation_quant(quantize_pointer f, float *d, uint4x4_t *q_answer){
    uint4x4_t *q = (uint4x4_t*)malloc(sizeof(uint4x4_t) * FEATURES/2 * CLASSES/2);
	float min, max;
	f(d, q, &min, &max, FEATURES, CLASSES);
	for (int i=0; i<FEATURES; i = i+2){
    	for (int j=0; j<CLASSES; j = j+2){
    		if (q[i/2 + j/2].i1 != q_answer[i/2 + j/2].i1){
    			printf("Error in quantized weight matrix at [%d][%d]\n\n", i, j);
    			printf("Expected value: %d\t Actual value: %d\n", q_answer[2*i/2 + j/2].i1, q[2*i/2 + j/2].i1);
    			return 0;
    		}
    		if (q[i/2 + j/2].i2 != q_answer[i/2 + j/2].i2){
    			printf("Error in quantized weight matrix at [%d][%d]\n\n", i, j +1);
    			printf("Expected value: %d\t Actual value: %d\n", q_answer[2*i/2 + j/2].i2	, q[2*i/2 + j/2].i2);

    			return 0;
    		}
		}
		for (int j=0; j<CLASSES; j = j+2){
			if (q[i/2 + j/2].i3 != q_answer[i/2 + j/2].i3){
    			printf("Error in quantized weight matrix at [%d][%d]\n\n", i + 1, j);
    			printf("Expected value: %d\t Actual value: %d\n", q_answer[2*i/2 + j/2].i3	, q[2*i/2 + j/2].i3);
    			
    			return 0;
    		}
    		if (q[i/2 + j/2].i4 != q_answer[i/2 + j/2].i4){
    			printf("Error in quantized weight matrix at [%d][%d]\n\n", i + 1, j +1);
    			printf("Expected value: %d\t Actual value: %d\n", q_answer[2*i/2 + j/2].i4	, q[2*i/2 + j/2].i4	);
    			
    			return 0;
    		}
		}
	}
printf("test Passed!!\n" );

return 1;
}


/*
Validation framework for vanilla implentation
*/
int validation_qmm4x4(qmm_pointer_4x4 f,float l_scale, float r_scale, float result_scale, uint4x4_t l_offset, uint4x4_t r_offset, 
	uint4x4_t result_offset, uint4x4_t* l_int_mat, uint4x4_t* r_int_mat, uint4x4_t* result_array, 
	int n, int k, int m){

    uint4x4_t *result_q = (uint4x4_t*)malloc(sizeof(uint4x4_t) * TEST_POINTS/2 * CLASSES/2);
    
    f(l_scale, r_scale, result_scale, l_offset, r_offset,
    result_offset, l_int_mat, r_int_mat, result_q, n, k, m);

    for (int i=0; i<n; i = i+2){
    	for (int j=0; j<m; j = j+2){
    		
    		if (result_q[2*i/2 + j/2].i1 != result_array[2*i/2 + j/2].i1){
    			printf("Error in quantized matrix at [%d][%d]\n\n", i, j);
    			printf("Expected value: %u\t Actual value: %u\n",  result_array[2*i/2 + j/2].i1, result_q[2*i/2 + j/2].i1 );
    			return 0;
    		}

    		if (result_q[2*i/2 + j/2].i2 != result_array[2*i/2 + j/2].i2){
    			printf("Error in quantized matrix at [%d][%d]\n\n", i, j +1);
    			printf("Expected value: %u\t Actual value: %u\n",  result_array[2*i/2 + j/2].i2, result_q[2*i/2 + j/2].i2 );
    			return 0;
    		}

		}
		for (int j=0; j<m; j = j+2){

			if (result_q[2*i/2 + j/2].i3 != result_array[2*i/2 + j/2].i3){
    			printf("Error in quantized matrix at [%d][%d]\n\n", i, j);
    			printf("Expected value: %u\t Actual value: %u\n",  result_array[2*i/2 + j/2].i3, result_q[2*i/2 + j/2].i3 );
    			return 0;
    		}

    		if (result_q[2*i/2 + j/2].i4 != result_array[2*i/2 + j/2].i4){
    			printf("Error in quantized matrix at [%d][%d]\n\n", i + 1, j +1);
    			printf("Expected value: %u\t Actual value: %u\n",  result_array[2*i/2 + j/2].i4, result_q[2*i/2 + j/2].i4 );
    			return 0;
    		}
		}
	}

    
	printf("test Passed!!\n" );

	return 1;
}