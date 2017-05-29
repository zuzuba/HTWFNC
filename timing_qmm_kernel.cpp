#include "utils.h"
#include "qmm.h"
#include "quantize.h"
#include "qmm_kernel.h"
#include "trick_vector.h"
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
typedef void(*qmm_kernel_pointer)(uint4x4_t* , uint4x4_t* ,int16_t* ,int ,int ,int );

double perf_test(qmm_kernel_pointer f, char *desc,int dim);

#define TEST_FAILED -1

#define MAX_FUNCS 32


void register_functions_4x4();
void add_function_4x4(qmm_kernel_pointer f, char *name, int flop);
int validation_4x4(qmm_kernel_pointer	 f);


/* Global vars, used to keep track of student functions */

qmm_kernel_pointer userFuncs_4x4[MAX_FUNCS];
char *funcNames_4x4[MAX_FUNCS];
int Flops[MAX_FUNCS];
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
	add_function_4x4(&qmm_kernel_trick, (char *)"naive_trick",4);
	add_function_4x4(&qmm_kernel_trick_blocking, (char *)"trick_blocking",2);
	add_function_4x4(&qmm_kernel_trick_AVX, (char *)"trick_AVX",2);
	add_function_4x4(&qmm_kernel_trick_AVX_unrolled, (char *)"AVX_unrolled",2);
	// Add your functions here
	// add_function(&your_function, "function: Optimization X", nrflops);
	
}

/*
* Registers a user function to be tested by the driver program. Registers a
* string description of the function as well
*/


void add_function_4x4(qmm_kernel_pointer f, char *name, int flops)
{	
	if (numFuncs_4x4 >= MAX_FUNCS)
	{
		printf("Couldn't register %s, too many functions registered (Max: %d)",
			name, MAX_FUNCS);
		return;
	}

	userFuncs_4x4[numFuncs_4x4] = f;
	funcNames_4x4[numFuncs_4x4] = name;
	Flops[numFuncs_4x4] = flops;

	numFuncs_4x4++;
}


int main(int argc, char **argv)
//int test_qmm()
{
	printf("------Timing qmm_kernel-----\n");
	int verbosity = 2;
    float cycles,perf;
    char file_name[60], func_name[60],file_name_cycles[60];
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
		printf("Performance of qmm_kernel function: %s \n", funcNames_4x4[i]);
		strcpy(func_name, funcNames_4x4[i]);
		strcpy(file_name,"data/perf_qmm_kernel_");
		strcpy(file_name_cycles,"data/cycles_qmm_kernel_");
		strcat(file_name, func_name);
		strcat(file_name, ".dat");
		strcat(file_name_cycles, func_name);
		strcat(file_name_cycles, ".dat");
		FILE *fp = fopen(file_name,"w+");
		FILE *fp_cycles = fopen(file_name_cycles,"w+");
		for(int n=30; n<400;n+=30){
			cycles = perf_test(userFuncs_4x4[i],funcNames_4x4[i],n);
			perf = (Flops[i]*n*n*n )/cycles;
			printf("%s: n:%d cycles:%f perf:%f \n",funcNames_4x4[i],n, cycles,perf);
			fprintf(fp, "%d %f\n",n,perf);
			fprintf(fp_cycles, "%d %f\n",n,cycles);
		}	
		fclose(fp);
		fclose(fp_cycles);
	}

	return 0;
}


double perf_test(qmm_kernel_pointer f, char *desc,int n)
{
	double cycles = 0.;
	double perf = 0.0;
	long num_runs = 100;
	double multiplier = 1;
	myInt64 start, end;

	float *lhs,*rhs;
	uint4x4_t *lhs_q,*rhs_q,*result_q;
	float lhs_mn, lhs_mx, rhs_mn, rhs_mx;
	float lhs_scale,rhs_scale,lhs_zero_point,rhs_zero_point;
	uint16_t term4;

	lhs = build_full_mat(n);
	rhs = build_full_mat(n);
	lhs_q = allocate_quantized_mat(n);
	rhs_q = allocate_quantized_mat(n);

	int16_t *acc = (int16_t *)malloc(sizeof(int16_t)*n*n); 
	for (int i = 0; i < n*n; ++i) acc[i]=0;
	
	quantize_4x4(lhs, lhs_q, &lhs_mn , &lhs_mx, n, n);
	quantize_parameter(lhs_mn, lhs_mx,  &lhs_scale, &lhs_zero_point);
	quantize_4x4(rhs, rhs_q, &rhs_mn , &rhs_mx, n, n);
	quantize_parameter(rhs_mn, rhs_mx,  &rhs_scale, &rhs_zero_point);

	//
	uint4x4_t l_offset;
	l_offset.i1= (int)lhs_zero_point;
	uint4x4_t r_offset;
	r_offset.i1= (int)rhs_zero_point;

	n=n/2;

	// Warm-up phase: we determine a number of executions that allows
	// the code to be executed for at least CYCLES_REQUIRED cycles.
	// This helps excluding timing overhead when measuring small runtimes.
	do {
		num_runs = num_runs * multiplier;
		start = start_tsc();
		for (size_t i = 0; i < num_runs; i++) {
			f(lhs_q,rhs_q,acc,n,n,n);			
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
			f(lhs_q,rhs_q,acc,n,n,n);
		}
		end = stop_tsc(start);

		cycles = ((double)end) / num_runs;

		cyclesList.push_back(cycles);
	}

	
	free(acc);
	free(rhs_q);
	free(lhs_q);
	free(rhs);
	free(lhs);

	cyclesList.sort();
	cycles = cyclesList.front();

	return cycles;
}