#include "utils.h"
#include "naive_qmm.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TEST_FAILED 0

/* prototype of the function you need to optimize */
typedef void(*qmm_pointer)(float, float, float, uint4x1_t, uint4x1_t, uint4x1_t, 
	uint4x1_t*, uint4x1_t*, uint4x1_t*, int , int, int);

typedef void(*qmm_pointer_4x4)(float, float, float, uint4x4_t, uint4x4_t, uint4x4_t, 
	uint4x4_t*, uint4x4_t*, uint4x4_t*, int , int, int);

#define MAX_FUNCS 32

void register_functions();
void add_function(qmm_pointer f, char *name, int flop);
int validation(qmm_pointer f);

void register_functions_4x4();
void add_function_4x4(qmm_pointer_4x4 f, char *name, int flop);
int validation_4x4(qmm_pointer_4x4	 f);


/* Global vars, used to keep track of student functions */
qmm_pointer userFuncs[MAX_FUNCS];
char *funcNames[MAX_FUNCS];
int funcFlops[MAX_FUNCS];
int numFuncs = 0;

qmm_pointer_4x4 userFuncs_4x4[MAX_FUNCS];
char *funcNames_4x4[MAX_FUNCS];
int funcFlops_4x4[MAX_FUNCS];
int numFuncs_4x4 = 0;


void register_functions()
{	
	add_function(&qmm_space_waste, (char *)"naive",7);
	// Add your functions here
	// add_function(&your_function, "function: Optimization X", nrflops);
	
}

void register_functions_4x4()
{	
	add_function_4x4(&qmm_naive, (char *)"naive 4x4",7);
	// Add your functions here
	// add_function(&your_function, "function: Optimization X", nrflops);
	
}

/*
* Registers a user function to be tested by the driver program. Registers a
* string description of the function as well
*/
void add_function(qmm_pointer f, char *name, int flops)
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


//int main(int argc, char **argv)
int test_qmm()
{
	printf("Starting program\n");
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
    

    // 4x4 section
    // Initialize the vectors of functions, function names and function flops
	register_functions_4x4();
    
    // Message if there are zero functions
	if (numFuncs_4x4 == 0)
	{
		printf("No 4x4 functions registered - nothing for driver to do\n");
		printf("Register functions by calling register_func(f, name)\n");
		printf("in register_funcs()\n");

		return TEST_FAILED;
	}
	printf("\n%d 4x4 functions registered\n", numFuncs_4x4);

	// Test of vanilla implementation first
	test_success = validation_4x4(userFuncs_4x4[0]);
	if (test_success == 0)
	{
		printf("Vanilla 4x4 implementation failed test!!\n");
		return TEST_FAILED;
	}
	return 1;
}



/*
Validation framework for vanilla implentation
*/
int validation(qmm_pointer f){

	/* Test input that does not need offset to represent 0 exactly
    Min = -10, Max = 15.5, delta = 0, no offset
    */
    // float l_float[3][2] = {-1, -0.89,
    // 					   -0.79, -0.69,
    // 					   -0.59, 0.5};

    // float r_float[2][3] = {- 1, -0.39, 0.11,
    // 						-0.49, -0.29, 0.5};

    float l_float[3][2] = {-1, -0.9,
    					   -0.8, -0.7,
    					   -0.6, 0.5};

    float r_float[2][3] = {- 1, -0.4, 0.11,
    						-0.5, -0.3, 0.5};


	uint8_t l_array[3][2] = {0, 1,
							 2, 3,
							 4, 15};

	uint8_t r_array[2][3] = {0, 6, 11,
							 5, 7, 15};

	uint8_t result_array[3][3] = {15, 15, 4,
								  15, 15, 6,
								  14, 11, 12};

    uint4x1_t *l_mat = (uint4x1_t*)malloc(sizeof(uint4x1_t) * 6);
    uint4x1_t *r_mat = (uint4x1_t*)malloc(sizeof(uint4x1_t) * 6);;
    uint4x1_t *result_mat = (uint4x1_t*)malloc(sizeof(uint4x1_t) * 9);;

    float l_scale = 0.1;
    float r_scale = 0.1;
    float result_scale = 0.1;

    uint4x1_t l_offset, r_offset, result_offset;
    l_offset.i = 10;
    r_offset.i = 10;
    result_offset.i = 10;

    for (int i=0; i<3; i++){
    	for (int j=0; j<2; j++){
    		(l_mat[2*i + j]).i = l_array[i][j];
    	}
    }
    
    for (int i=0; i<2; i++){
    	for (int j=0; j<3; j++){
    		(r_mat[3*i + j]).i = r_array[i][j];
    	}
    }

    f(l_scale, r_scale, result_scale, l_offset, r_offset,
    result_offset, l_mat, r_mat, result_mat, 3, 2, 3);

    for (int i=0; i<3; i++){
    	for (int j=0; j<3; j++){
    		if (result_array[i][j] != (result_mat[3*i + j]).i){
    			printf("Error in quantized matrix at [%d][%d]\n\n", i, j);
				printf("Expected value: %u\t Actual value: %u\n",  result_array[i][j], (result_mat[3*i + j]).i);
				return 0;
    		}
		}
	}

    
	printf("First test Passed!!\n" );

	return 1;
}


/*
Validation framework for vanilla implentation
*/
int validation_4x4(qmm_pointer_4x4 f){

	/* Test input that does not need offset to represent 0 exactly
    Min = -10, Max = 15.5, delta = 0, no offset
    */
    // float l_float[3][2] = {-1, -0.89,
    // 					   -0.79, -0.69,
    // 					   -0.59, 0.5};

    // float r_float[2][3] = {- 1, -0.39, 0.11,
    // 						-0.49, -0.29, 0.5};

    float l_float[4][2] = {-1, -0.9,
    					   -0.8, -0.7,
    					   -0.6, -0.5,
    					   -0.4, 0.5};

    float r_float[2][4] = {- 1, -0.4, 0.1, 0.3,
    						-0.5, -0.3, 0.2, 0.5};


	uint8_t l_array[4][2] = {0, 1,
							 2, 3,
							 4, 5,
							 6, 15};

	uint8_t r_array[2][4] = {0, 6, 11, 13,
							 5, 7, 12, 15};

	uint8_t result_array[4][4] = {15, 15, 7, 2,
								  15, 15, 8, 4,
								  15, 14, 8, 6,
								  12, 10, 11, 11};

    uint4x4_t *l_mat = (uint4x4_t*)malloc(sizeof(uint4x4_t) * 2);
    uint4x4_t *r_mat = (uint4x4_t*)malloc(sizeof(uint4x4_t) * 2);;
    uint4x4_t *result_mat = (uint4x4_t*)malloc(sizeof(uint4x4_t) * 4);;

    float l_scale = 0.1;
    float r_scale = 0.1;
    float result_scale = 0.1;

    uint4x4_t l_offset, r_offset, result_offset;
    l_offset.i1 = 10;
    r_offset.i1 = 10;
    result_offset.i1 = 10;

    for (int i=0; i<2; i = i+1){
    	l_mat[i].i1 = l_array[i*2][0];
    	l_mat[i].i2 = l_array[i*2][1];
    	l_mat[i].i3 = l_array[i*2 + 1][0];
    	l_mat[i].i4 = l_array[i*2 + 1][1];
    }

    for (int i=0; i<2; i = i+1){
 	    r_mat[i].i1 = r_array[0][2*i];
    	r_mat[i].i2 = r_array[0][2*i + 1];
    	r_mat[i].i3 = r_array[1][2*i];
    	r_mat[i].i4 = r_array[1][2*i + 1];	
    }
    
    
    f(l_scale, r_scale, result_scale, l_offset, r_offset,
    result_offset, l_mat, r_mat, result_mat, 4, 2, 4);

    for (int i=0; i<4; i = i+2){
    	for (int j=0; j<4; j = j+2){
    		
    		if (result_mat[2*i/2 + j/2].i1 != result_array[i][j]){
    			printf("Error in quantized matrix at [%d][%d]\n\n", i, j);
    			return 0;
    		}

    		if (result_mat[2*i/2 + j/2].i2 != result_array[i][j +1]){
    			printf("Error in quantized matrix at [%d][%d]\n\n", i, j +1);
    			return 0;
    		}

		}
		for (int j=0; j<4; j = j+2){

			if (result_mat[2*i/2 + j/2].i3 != result_array[i+1][j]){
    			printf("Error in quantized matrix at [%d][%d]\n\n", i, j);
    			return 0;
    		}

    		if (result_mat[2*i/2 + j/2].i4 != result_array[i+1][j +1]){
    			printf("Error in quantized matrix at [%d][%d]\n\n", i + 1, j +1);
    			return 0;
    		}
		}
	}

    
	printf("First test Passed!!\n" );

	return 1;
}