#include "utils.h"
#include "quantize.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#define TEST_FAILED -1


/* prototype of the function you need to optimize */
typedef void(*quantize_pointer)(float *, uint4x4_t*,  float *, float *, int, int);

#define MAX_FUNCS 32

void register_functions();
void add_function(quantize_pointer f, char *name, int flop);
int validation(quantize_pointer f);


/* Global vars, used to keep track of student functions */
quantize_pointer userFuncs[MAX_FUNCS];
char *funcNames[MAX_FUNCS];
int funcFlops[MAX_FUNCS];
int numFuncs = 0;


void register_functions()
{	
	add_function(&quantize_4x4, (char *)"naive",7);
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


//int main(int argc, char **argv)
int main()
{
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
	printf("\n%d functions registered\n", numFuncs);

	// Test of vanilla implementation first
	test_success = validation(userFuncs[0]);
	if (test_success == 0)
	{
		printf("Vanilla implementation failed test!!\n");
		return TEST_FAILED;
	}
	return 0;

}



int validation(quantize_pointer f){

	float d_array[4][2] = {-0.81,   -1, 
					        0.22, 0.36,
					        -0.3, 0.05,
					        0.5, -0.47};

	uint8_t q_answer[4][2] = {2,   0,
							  12, 14,
							   7, 11,
							  15,  5};

	float *d = (float*)malloc(sizeof(float) * 8);
	uint4x4_t *q = (uint4x4_t *)malloc(sizeof(uint4x4_t) * 2);

	float min, max;
	int rows = 4;
	int columns = 2;

	for (int i=0; i<rows; i++){
		for(int j = 0; j<columns; j++){
			d[i*columns + j] = d_array[i][j];
		}
	}


	f(d, q, &min, &max, rows, columns);


	for (int i=0; i<rows; i = i+2){
    	for (int j=0; j<columns; j = j+2){

    		if (q[i/2 + j/2].i1 != q_answer[i][j]){
    			printf("Error in quantized matrix at [%d][%d]\n\n", i, j);
    			printf("Expected value: %d\t Actual value: %d\n", q_answer[i][j], q[2*i/2 + j/2].i1);
    			return 0;
    		}

    		if (q[i/2 + j/2].i2 != q_answer[i][j +1]){
    			printf("Error in quantized matrix at [%d][%d]\n\n", i, j +1);
    			printf("Expected value: %d\t Actual value: %d\n", q_answer[i][j +1], q[2*i/2 + j/2].i2);

    			return 0;
    		}

		}
		for (int j=0; j<columns; j = j+2){

			if (q[i/2 + j/2].i3 != q_answer[i+1][j]){
    			printf("Error in quantized matrix at [%d][%d]\n\n", i + 1, j);
    			printf("Expected value: %d\t Actual value: %d\n", q_answer[i+1][j], q[2*i/2 + j/2].i3);
    			
    			return 0;
    		}

    		if (q[i/2 + j/2].i4 != q_answer[i+1][j +1]){
    			printf("Error in quantized matrix at [%d][%d]\n\n", i + 1, j +1);
    			printf("Expected value: %d\t Actual value: %d\n", q_answer[i+1][j+1], q[2*i/2 + j/2].i4	);
    			
    			return 0;
    		}


		}
	}
printf("First test Passed!!\n" );

return 1;
}


