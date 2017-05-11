/**
*      _________   _____________________  ____  ______
*     / ____/   | / ___/_  __/ ____/ __ \/ __ \/ ____/
*    / /_  / /| | \__ \ / / / /   / / / / / / / __/
*   / __/ / ___ |___/ // / / /___/ /_/ / /_/ / /___
*  /_/   /_/  |_/____//_/  \____/\____/_____/_____/
*
*  http://www.inf.ethz.ch/personal/markusp/teaching/
*  How to Write Fast Numerical Code 263-2300 - ETH Zurich
*  Copyright (C) 2017  Alen Stojanov      (astojanov@inf.ethz.ch)
*                      Georg Ofenbeck     (ofenbeck@inf.ethz.ch)
*                      Singh Gagandeep    (gsingh@inf.ethz.ch)
*	                Markus Pueschel    (pueschel@inf.ethz.ch)
*
*  This program is free software: you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation, either version 3 of the License, or
*  (at your option) any later version.
*
*  This program is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this program. If not, see http://www.gnu.org/licenses/.
*/
//#include "stdafx.h"


#include <list>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <string.h>
#include "tsc_x86.h"
#include <stdint.h>

#include "naiv.h"


/* prototype of the function you need to optimize */
typedef void(*quant)(double**, uint8_t**, double*, double*, int);



#define CYCLES_REQUIRED 1e7
#define REP 30
#define MAX_FUNCS 32
#define FLOPS (4.*n)
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




void add_function(quant f, char *name, int flop);

/* Global vars, used to keep track of student functions */
quant userFuncs[MAX_FUNCS];
char *funcNames[MAX_FUNCS];
int funcFlops[MAX_FUNCS];
int numFuncs = 0;


void rands(double * m, size_t row, size_t col)
{/* Initialize matrix at random*/
	for (size_t i = 0; i < row*col; ++i)  
		m[i] = static_cast<double>(rand() + 1) / RAND_MAX;
}

double** build_full_mat(unsigned n)
{/* Allocate memory and initialize at random*/
	int i;
	double **d;
	d  = (double **)malloc(sizeof(double *) * n);
    d[0] = (double *)malloc(sizeof(double) * n * n);
 
    for(i = 0; i < n; i++){
        d[i] = (*d + n * i);
		rands(d[i], 1, n);
	}
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

void destroy(double * m)
{/* Free memory*/
	free(m);
}

/*
* Called by the driver to register your functions
* Use add_function(func, description) to add your own functions
*/
void register_functions()
{	
	add_function(&vanilla_quantize, (char *)"naive",7);
	// Add your functions here
	// add_function(&your_function, "function: Optimization X", nrflops);
	
}


/*
* Main driver routine - calls register_funcs to get student functions, then
* tests all functions registered, and reports the best performance
*/
int main(int argc, char **argv)
{
	printf("Starting program\n");
	double perf;
	double maxPerf = 0;
	double clocksElapsed, timeElapsed;
	int i;
	int maxInd = 0;
	int verbosity = 2;
	int test_success = 0;
	FILE *fp;
    
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
    
	
    // Init vectors at random for computation
	double **d;
	uint8_t **q;
	double mn, mx;
	int n = 128;

	d = build_full_mat(n);
	q = allocate_quantized_mat(n);

    // Get result of slow implementation
	quant f = userFuncs[0];
	f(d,q,&mn,&mx,n);
	//double result = z[0];
	double error = 0;

    // Make sure the result of optimized implementation is correct
	for (i = 0; i < numFuncs; i++)
    {
		quant f = userFuncs[i];
		f(d,q,&mn,&mx,n);
		//error = fabs(z[0] - result);
		//result = z[0];

        if (error > EPS)
			printf("ERROR!!!!  the results for the %d th function are different to the previous", i);
	}
    // Free memory
	free(d);
	free(q);

    const char perf_str[30]="_perf.dat";
    char fun_name[30];
    // Measure the performance of the different implementations
	for (i = 0; i < numFuncs; i++)
	{		
		strcpy(fun_name,funcNames[i]);
		fp = fopen(strcat(fun_name,perf_str),"w+");
		for(n=3;n<=200;n+=1){
			perf = perf_test(userFuncs[i], funcNames[i], funcFlops[i],n);
			printf("\nPerformance: %s\nPerf: %.5f FLOPs/c  Cycles: %.3f cycles\n", funcNames[i], perf, ((double)funcFlops[i])/perf);
			fprintf(fp, "%d %f \n",n,perf );
		}
		fclose(fp);
	}

	// Measure the elapsed time and clock ticks of each implementation
	for (i = 0; i < numFuncs; i++)
	{	
		clocksElapsed = c_clock(userFuncs[i]);
		timeElapsed = timeofday(userFuncs[i]);
		printf("\nTimeElapsed: %s\nMeasured by c_clock(): %lf seconds\nMeasured by timeofday(): %lf seconds\n", funcNames[i], clocksElapsed, timeElapsed);
	}


	return 0;
}




// Test for vanilla implementation
int validation(quant f){

	/* Test input that does not need offset to represent 0 exactly
    Min = -10, Max = 15.5, delta = 0, no offset
    */
    double d_test_array[3][3] = {-10.0,     0.0,       1.47,
                               -7.89,    3.05,    -9.9431,
                                15.3,   -10.0,       15.5};
    // Expected output
    uint8_t q_answer[3][3] = { 0,    100,    115,
                              21,    131,      1,
                             253,      0,    255};
    
    double **d_test;
	uint8_t **q_test;

    double mn, mx;
    int n =3;

    // Initialize values
    d_test = build_full_mat(n);
	q_test = allocate_quantized_mat(n);
	
	// Write test input to the allocated matrix
	for (int i=0; i<n; i++){
		for (int j=0; j<n; j++){
			d_test[i][j] = d_test_array[i][j];
		}
	}

	// Call function
	f(d_test, q_test, &mn, &mx, n);

	// Check output is correct
	for(int i=0; i<n; i++){
		for (int j=0; j<n; j++){
			if(q_test[i][j] != q_answer[i][j]){
				printf("Error in quantized matrix at [%d][%d]\n\n", i, j);
				printf("Expected value: %u\t Actual value: %u\n",  q_answer[i][j], q_test[i][j]);
				return 0;
			}
		}
	}

	printf("First test Passed!!\n" );
	//return 1;

	/* Test input
	Min = -10.03, Max = 15.47, delta = 0.1, need to offsey by 0.03
	Notice that entry [2][1] would corrspond to 1 if the shif was not performed correctly.
	*/
    double d_test_array1[3][3] = {-10.03,     0.0,       1.47,
                               -7.89,    3.05,    -9.9431,
                                15.3,   -9.97,       15.47};


	for (int i=0; i<n; i++){
		for (int j=0; j<n; j++){
			d_test[i][j] = d_test_array1[i][j];
		}
	}

	// Call function
	f(d_test, q_test, &mn, &mx, n);

	// Check output is correct
	for(int i=0; i<n; i++){
		for (int j=0; j<n; j++){
			if(q_test[i][j] != q_answer[i][j]){
				printf("Error in quantized matrix at [%d][%d]\n\n", i, j);
				printf("Expected value: %u\t Actual value: %u\n",  q_answer[i][j], q_test[i][j]);
				return 0;
			}
		}
	}

	printf("Second test Passed!!\n" );

		/* Test input
	Min = -9.96, Max = 15.54, delta = 0.1, need to offsey by -0.04
	Notice that entry [2][1] would corrspond to 254 if the shif was not performed correctly.
	*/
    double d_test_array2[3][3] = {-9.96,     0.0,       1.47,
                               -7.89,    3.05,    -9.9431,
                                15.3,   15.48,       15.54};
    q_answer[2][1] = 255;


	for (int i=0; i<n; i++){
		for (int j=0; j<n; j++){
			d_test[i][j] = d_test_array2[i][j];
		}
	}

	// Call function
	f(d_test, q_test, &mn, &mx, n);

	// Check output is correct
	for(int i=0; i<n; i++){
		for (int j=0; j<n; j++){
			if(q_test[i][j] != q_answer[i][j]){
				printf("Error in quantized matrix at [%d][%d]\n\n", i, j);
				printf("Expected value: %u\t Actual value: %u\n",  q_answer[i][j], q_test[i][j]);
				return 0;
			}
		}
	}

	//Reset the q_answer to initial value
	q_answer[2][1] = 0;

	printf("Third test Passed!!\n" );

	/* Test input - Matrix all positive and offset needs to be saturated
	Min = 1.0, Max = 26.5, delta = 0.1, need to shift by -1.0, which means to saturate the int_offset.
	*/
    double d_test_array3[3][3] = {1.0, 	14.23, 1.05,
    							  8.39, 16.00, 21.9,
    							  25.8, 25.5,  26.5};

   	uint8_t q_answer3[3][3] = {10, 142, 11,
   							   84, 160, 219,
   							   255, 255, 255};

	for (int i=0; i<n; i++){
		for (int j=0; j<n; j++){
			d_test[i][j] = d_test_array3[i][j];
		}
	}

	// Call function
	f(d_test, q_test, &mn, &mx, n);

	// Check output is correct
	for(int i=0; i<n; i++){
		for (int j=0; j<n; j++){
			if(q_test[i][j] != q_answer3[i][j]){
				printf("Error in quantized matrix at [%d][%d]\n\n", i, j);
				printf("Expected value: %u\t Actual value: %u\n",  q_answer3[i][j], q_test[i][j]);
				return 0;
			}
		}
	}

	printf("Fourth test Passed!!\n" );

		/* Test input - Matrix all positive and offset needs to be saturated
	Min = -26.5, Max = -1, delta = 0.1, need to shift by -1.0, which means to saturate the int_offset.
	*/
    double d_test_array4[3][3] = {-1.0, 	-14.23, -1.05,
    							  -8.39, -16.00, -21.9,
    							  -25.8, -25.5,  -26.5};

   	uint8_t q_answer4[3][3] = {245, 113, 245,
   							   171, 95, 36,
   							   0, 0, 0};

	for (int i=0; i<n; i++){
		for (int j=0; j<n; j++){
			d_test[i][j] = d_test_array4[i][j];
		}
	}

	// Call function
	f(d_test, q_test, &mn, &mx, n);

	// Check output is correct
	for(int i=0; i<n; i++){
		for (int j=0; j<n; j++){
			if(q_test[i][j] != q_answer4[i][j]){
				printf("Error in quantized matrix at [%d][%d]\n\n", i, j);
				printf("Expected value: %u\t Actual value: %u\n",  q_answer4[i][j], q_test[i][j]);
				return 0;
			}
		}
	}

	printf("Fifth test Passed!!\n" );

	return 1;
}




/*
* Registers a user function to be tested by the driver program. Registers a
* string description of the function as well
*/
void add_function(quant f, char *name, int flops)
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




/*
* Checks the given function for validity. If valid, then computes and
* reports and returns its performance in MFLOPs
*/
double perf_test(quant f, char *desc, int flops,int n)
{
	double cycles = 0.;
	double perf = 0.0;
	long num_runs = 100;
	double multiplier = 1;
	myInt64 start, end;

	double **d;
	uint8_t **q;
	double mn, mx;

	d = build_full_mat(n);
	q = allocate_quantized_mat(n);
	

	// Warm-up phase: we determine a number of executions that allows
	// the code to be executed for at least CYCLES_REQUIRED cycles.
	// This helps excluding timing overhead when measuring small runtimes.
	do {
		num_runs = num_runs * multiplier;
		start = start_tsc();
		for (size_t i = 0; i < num_runs; i++) {
			f(d,q,&mn,&mx,n);			
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
			f(d,q,&mn,&mx,n);
		}
		end = stop_tsc(start);

		cycles = ((double)end) / num_runs;

		cyclesList.push_back(cycles);
		perfList.push_back(FLOPS / cycles);
	}

	//printf("%f", y[0]);
	free(d);
	free(q);
	cyclesList.sort();
	cycles = cyclesList.front();

	return (n*n*flops*1.0) / cycles;
}

/* 
 * Timing function --- more accurate than rdtsc() on Unix Systems.
 * return time elapsed in seconds 
 */
double c_clock(quant f) {
	int i, num_runs;
	double cycles;
	clock_t start, end;
	// definition of parameters used in f
	double **d;
	uint8_t **q;
	double mn, mx;
	int n = 128;
	// initialization of parameters.
	d = build_full_mat(n);
	q = allocate_quantized_mat(n);
	// number of runs we measure
	num_runs = NUM_RUNS;
	// if calibrate, num_runs changes to 2 ^ 14
	#ifdef CALIBRATE
		while(num_runs < (1 << 14)) {
			start = clock();
			for (i = 0; i < num_runs; ++i) {
				f(d,q,&mn,&mx,n);
			}
			end = clock();
			cycles = (double)(end-start);
			// Same as in c_clock: CYCLES_REQUIRED should be expressed accordingly to the order of magnitude of CLOCKS_PER_SEC
			if(cycles >= CYCLES_REQUIRED/(FREQUENCY/CLOCKS_PER_SEC)) break;
			num_runs *= 2;
		}
	#endif
	// actual timing starts here
	start = clock();
	for(i=0; i<num_runs; ++i) {
		// we run the quantization code.
		f(d,q,&mn,&mx,n); 
	}
	end = clock();
	free(d);
	free(q);

	return (double)((end-start)/num_runs)/CLOCKS_PER_SEC;
}
/*
 * Measure the running time of function f ---- works on Unix Systems
 * return running time in second
 */
double timeofday(quant f) {
	int i, num_runs;
	double cycles;
	struct timeval start, end;
	// definition of parameters used in f
	double **d;
	uint8_t **q;
	double mn, mx;
	int n = 128;
	// initialization of parameters.
	d = build_full_mat(n);
	q = allocate_quantized_mat(n);
	// if calibrate, num_runs changes to 2 ^ 14
	num_runs = NUM_RUNS;
	#ifdef CALIBRATE
		while(num_runs < (1 << 14)) {
			gettimeofday(&start, NULL);
			for (i = 0; i < num_runs; ++i) {
				f(d,q,&mn,&mx,n);
			}
			gettimeofday(&end, NULL);
			cycles = (double)((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1e6)*FREQUENCY;
			if(cycles >= CYCLES_REQUIRED) break;
			num_runs *= 2;
		}
	#endif
	// start measure	
	gettimeofday(&start, NULL);
	for(i=0; i < num_runs; ++i) {
		f(d,q,&mn,&mx,n); 
	}
	gettimeofday(&end, NULL);
	free(d);
	free(q);
	return (double)((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1e6)/ num_runs;
}


