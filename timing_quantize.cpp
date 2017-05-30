#include "utils.h"
#include "qmm.h"
#include "quantize.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <list>
#include <time.h>
#include <sys/time.h>
#include "tsc_x86.h"

#define CYCLES_REQUIRED 1e7
#define REP 30
#define MAX_FUNCS 32
#define EPS (1e-3)

/* start ---definition relevant to timing */
#define NUM_RUNS 1				// number of runs we measure, then average
#define FREQUENCY 3.4e9			// need to change on different computers. This value is the actual CPU frequency
// #define CALIBRATE				// Whether calibrate before measuring.
/* end   ---definition relevant to timing */

using namespace std;

//headers

/* prototype of the function you need to optimize */
typedef void(*quantize_pointer)(float *, uint4x4_t*,  float *, float *, int, int);

double perf_test(quantize_pointer f, char *desc,int dim);

#define TEST_FAILED -1

#define MAX_FUNCS 32


void register_functions_4x4();
void add_function_4x4(quantize_pointer f, char *name, int flop_quad_term, int flop_linear_term);
int validation_4x4(quantize_pointer f);


/* Global vars, used to keep track of student functions */

quantize_pointer userFuncs_4x4[MAX_FUNCS];
char *funcNames_4x4[MAX_FUNCS];
int funcFlops_quad_term_4x4[MAX_FUNCS];
int funcFlops_linear_term_4x4[MAX_FUNCS];
int numFuncs_4x4 = 0;


void rands(float * m, size_t row, size_t col)
{/* Initialize matrix at random*/
	for (size_t i = 0; i < row*col; ++i)  
		m[i] = static_cast<float>(rand() + 1) / RAND_MAX;
}

float* build_full_mat(unsigned n)
{/* Allocate memory and initialize at random*/
	int i;
	float *d;
	d  = (float *)malloc(sizeof(float ) * n*n);
 	
 	rands(d,n,n);
	return d;
}

uint4x4_t* allocate_quantized_mat(unsigned n){
	int i;
	uint4x4_t* q;
	q  = (uint4x4_t *)malloc(sizeof(uint4x4_t ) * n*n);
    return q;
}

void destroy(float * m)
{/* Free memory*/
	free(m);
}


void register_functions_4x4()
{	
	add_function_4x4(&quantize_4x4, (char *)"naive",7,0);
	add_function_4x4(&quantize_AVX, (char *)"AVX",7,0);
	
}

/*
* Registers a user function to be tested by the driver program. Registers a
* string description of the function as well
*/


void add_function_4x4(quantize_pointer f, char *name, int flop_quad_term, int flop_linear_term)
{	
	if (numFuncs_4x4 >= MAX_FUNCS)
	{
		printf("Couldn't register %s, too many functions registered (Max: %d)",
			name, MAX_FUNCS);
		return;
	}

	userFuncs_4x4[numFuncs_4x4] = f;
	funcNames_4x4[numFuncs_4x4] = name;
	funcFlops_quad_term_4x4[numFuncs_4x4] = flop_quad_term;
	funcFlops_linear_term_4x4[numFuncs_4x4] = flop_linear_term;

	numFuncs_4x4++;
}


int main(int argc, char **argv)
//int test_qmm()
{
	printf("------Timing quantization function-----\n");
	int verbosity = 2;
    float cycles, perf;
    char file_name[60],func_name[60],file_name_cycles[60];
    // Initialize the vectors of functions, function names and function flops
	// Test of vanilla implementation first
    

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

	for (int i = 0; i < numFuncs_4x4; ++i)
	{
		printf("Performance of quantize function: %s \n", funcNames_4x4[i]);
		strcpy(func_name, funcNames_4x4[i]);
		strcpy(file_name,"data/perf_quantize_");
		strcpy(file_name_cycles,"data/cycles_quantize_");
		strcat(file_name, func_name);
		strcat(file_name, ".dat");
		strcat(file_name_cycles, func_name);
		strcat(file_name_cycles, ".dat");
		FILE *fp = fopen(file_name,"w+");
		FILE *fp_cycles = fopen(file_name_cycles,"w+");

		for(int n=30; n<=400;n+=30){
			cycles = perf_test(userFuncs_4x4[i],funcNames_4x4[i],n);
			perf = (funcFlops_quad_term_4x4[i]*n*n + funcFlops_linear_term_4x4[i]*n)/cycles;
			printf("%s: n:%d  cycles:%f perf:%f \n",funcNames_4x4[i],n, cycles,perf);
			fprintf(fp, "%d %f\n",n,perf);
			fprintf(fp_cycles, "%d %f\n",n,cycles);
		}	
		fclose(fp);
		fclose(fp_cycles);
	}

	return 0;
}


double perf_test(quantize_pointer f, char *desc,int n)
{
	double cycles = 0.;
	double perf = 0.0;
	long num_runs = 100;
	double multiplier = 1;
	myInt64 start, end;

	float *d;
	uint4x4_t *q;
	float min, max;
	
	d = build_full_mat(n);
	q = allocate_quantized_mat(n);
	

	// Warm-up phase: we determine a number of executions that allows
	// the code to be executed for at least CYCLES_REQUIRED cycles.
	// This helps excluding timing overhead when measuring small runtimes.
	do {
		num_runs = num_runs * multiplier;
		start = start_tsc();
		for (size_t i = 0; i < num_runs; i++) {
			f(d, q, &min, &max, n, n);			
		}
		end = stop_tsc(start);

		cycles = (double)end;
		multiplier = (CYCLES_REQUIRED) / (cycles);
		
	} while (multiplier > 2);

	list< double > cyclesList;

	// Actual performance measurements repeated REP times.
	// We simply store all results and compute medians during post-processing.
	for (size_t j = 0; j < REP; j++) {

		start = start_tsc();
		for (size_t i = 0; i < num_runs; ++i) {
			f(d, q, &min, &max, n, n);			
		}
		end = stop_tsc(start);

		cycles = ((double)end) / num_runs;

		cyclesList.push_back(cycles);
	}

	free(d);
	free(q);

	cyclesList.sort();
	cycles = cyclesList.front();

	return cycles;
}