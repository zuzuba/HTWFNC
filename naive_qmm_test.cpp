/* prototype of the function you need to optimize */
typedef void(*qmm_pointer)(float, float, float, unit4x1_t, unit4x1_t, unit4x1_t, 
	unit4x1_t*, unit4x1_t*, unit4x1_t*, int , int, int);

#define MAX_FUNCS 32

void register_functions();
void add_function(qmm_pointer f, char *name, int flop);
int validation(qmm_pointer f)



/* Global vars, used to keep track of student functions */
qmm_pointer userFuncs[MAX_FUNCS];
char *funcNames[MAX_FUNCS];
int funcFlops[MAX_FUNCS];
int numFuncs = 0;


void register_functions()
{	
	add_function(&qmm_space_waste, (char *)"naive",7);
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



int main(int argc, char **argv)
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

		return 0;
	}
	printf("\n%d functions registered\n", numFuncs);

	// Test of vanilla implementation first
	test_success = validation(userFuncs[0]);
	if (test_success == 0)
	{
		printf("Vanilla implementation failed test!!\n");
		return 0;
	}
    
	return 0;
}



/*
Validation framework for vanilla implentation
*/
int validation(qmm_pointer f){

	/* Test input that does not need offset to represent 0 exactly
    Min = -10, Max = 15.5, delta = 0, no offset
    */
    float l_float[3][2] = {-1, -0.89,
    					   -0.79, -0.69,
    					   -0.59, 0.5};

    float r_float[2][3] = {- 1, -0.39, 0.11,
    						-0.49, -0.29, 0.5};

    float result_float[3][3];

	for (int i=0; i<3; i++){
		for (int j=0; j<3; j++){
			result_float[i][j] = 0;
			for (int k=0; k<2; k++){
				result_float[i][j] += l_float[i][k] * r_float[k][j];
			}
		}
	}

	uint8_t l_array[3][2] = {0, 1,
							 2, 3,
							 4, 14};

	uint8_t r_array[2][3] = {0, 6, 11,
							 5, 7, 15};

	uint8_t result_array[3][3] = {15, 15, 4,
								  15, 15, 5,
								  13, 10, 11};

    unit4x1_t *l_mat = (unit4x1_t*)malloc(sizeof(unit4x1_t * 6));
    unit4x1_t *r_mar = (unit4x1_t*)malloc(sizeof(unit4x1_t * 6));;
    unit4x1_t *result_mat = (unit4x1_t*)malloc(sizeof(unit4x1_t * 9));;

    float l_scale = 0.1;
    float r_scale = 0.1;
    float result_scale = 0.1;

    unit4x1_t l_offset, r_offset, result_offset;
    l_offset.i = 10;
    r_offset.i = 10;
    result_offset.i = 10;

    for (int i=0; i<3; i++){
    	for (int j=0; j<2; j++){
    		l_mat[2*i + j] = l_array[i][j];
    	}
    }
    
    for (int i=0; i<2; i++){
    	for (int j=0; j<3; j++){
    		r_mat[3*i + j] = r_array[i][j];
    	}
    }

    f(l_scale, r_scale, result_scale, l_offset, r_offset,
    result_offset, l_mat, r_mat, result_mat, 3, 2, 3);

    for (int i=0; i<3; i++){
    	for (int j=0; j<3; j++){
    		if (result_array[i][j] != result_mat[3*i + j]){
    			printf("Error in quantized matrix at [%d][%d]\n\n", i, j);
				printf("Expected value: %u\t Actual value: %u\n",  result_array[i][j], result_mat[3*i + j]);
				return 0;
    		}
		}
	}

    
	printf("First test Passed!!\n" );

	return 1;
}