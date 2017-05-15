#include "utils.h"
#include "naive_qmm.h"
#include <stdio.h>
#include <stdlib.h>
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
double get_perf_score(quant f);
void register_functions();
double perf_test(quant f, char *desc, int flops, int dim);
int validation(quant f);
double c_clock(quant f);
double timeofday(quant f);

#define TEST_FAILED -1

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

uint8_t** allocate_quantized_mat(unsigned n){
	int i;
	uint8_t **q;
	q  = (uint8_t **)malloc(sizeof(uint8_t *) * n);
    q[0] = (uint8_t *)malloc(sizeof(uint8_t) * n * n);
 
    for(i = 0; i < n; i++)
        q[i] = (*q + n * i);
    return q;
}

void destroy(float * m)
{/* Free memory*/
	free(m);
}



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


int main(int argc, char **argv)
//int test_qmm()
{
	printf("------Timing qmm-----\n");
	int verbosity = 2;
    float perf;
    FILE *fp = fopen("/data/perf_qmm.dat","w+");
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
	printf("Performance of qmm:\n");
	for(int n=100; n<2000;n+=100){
		perf = perf_test(userFuncs_4x4[0],funcNames_4x4[0],funcFlops_4x4[0],n);
		printf("%s: %f \n",funcNames[0],n, perf);
		fprintf(fp, "%d %f\n",n ,perf);
	}

	return 0;
}


double perf_test(qmm_pointer_4x4 f, char *desc, int flops,int n)
{
	double cycles = 0.;
	double perf = 0.0;
	long num_runs = 100;
	double multiplier = 1;
	myInt64 start, end;

	float *lhs,*rhs;
	uint4x4_t *lhs_q,*rhs_q;
	float lhs_mn, lhs_mx, rhs_mn, rhs_mx;
	float lhs_scale,rhs_scale,lhs_offset,rhs_offset;
	uint4x4_t lhs_offset, rhs_offset;

	lhs = build_full_mat(n);
	rhs = build_full_mat(n);
	lhs_q = allocate_quantized_mat(n);
	rhs_q = allocate_quantized_mat(n);
	result_q = allocate_quantized_mat(n);
	
	quantize_4x4(lhs, lhs_q, &mn_lhs , &mx_lhs, n, n);
	quantize_parameter(lhs_mn, lhs_mx,  &lhs_scale, &lhs_zero_point);
	quantize_4x4(rhs, rhs_q, &mn_rhs , &mx_rhs, n, n);
	quantize_parameter(rhs_mn, rhs_mx,  &rhs_scale, &rhs_zero_point);

	//
	uint4x4_t l_offset;
	l_offset.i1= (int)lhs_zero_point;
	uint4x4_t r_offset;
	r_offset.i1= (int)rhs_zero_point;
	

	// Warm-up phase: we determine a number of executions that allows
	// the code to be executed for at least CYCLES_REQUIRED cycles.
	// This helps excluding timing overhead when measuring small runtimes.
	do {
		num_runs = num_runs * multiplier;
		start = start_tsc();
		for (size_t i = 0; i < num_runs; i++) {
			f(scale_lhs, scale_rhs, scale_lhs, l_offset,  r_offset, l_offset, lhs_q, rhs_q, result_q,n,n,n);			
		}
		end = stop_tsc(start);

		cycles = (double)end;
		multiplier = (CYCLES_REQUIRED) / (cycles);
		
	} while (multiplier > 2);

	list< double > cyclesList, perfList;

	// Actual performance measurements repeated REP times.
	// We simply store all results and compute medians during post-processing.
	for (size_t j = 0; j < REP; j++) {

		start = start_tsc();
		for (size_t i = 0; i < num_runs; ++i) {
			f(scale_lhs, scale_rhs, scale_lhs, l_offset,  r_offset, l_offset, lhs_q, rhs_q, result_q,n,n,n);
		}
		end = stop_tsc(start);

		cycles = ((double)end) / num_runs;

		cyclesList.push_back(cycles);
		perfList.push_back(FLOPS / cycles);
	}

	//printf("%f", y[0]);
	free(lhs);
	free(rhs);
	free(lhs_q);
	free(rhs_q);
	
	cyclesList.sort();
	cycles = cyclesList.front();

	return (n*n*flops*1.0) / cycles;
}