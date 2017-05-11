/* prototype of the function you need to optimize */
typedef void(*qmm_pointer)(float, float, float, unit4x1_t, unit4x1_t, unit4x1_t, 
	unit4x1_t*, unit4x1_t*, unit4x1_t*, int , int, int);

#define MAX_FUNCS 32

void register_functions();
void add_function(qmm_pointer f, char *name, int flop);



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