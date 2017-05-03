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

using namespace std;

//headers
double get_perf_score(quant f);
void register_functions();
double perf_test(quant f, char *desc, int flops);
int validation(quant f);


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
	add_function(&vanilla_quantize, "Naive implementation",40/5);
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
	int i;
	int maxInd = 0;
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

    
    // Measure the performance of the different implementations
	for (i = 0; i < numFuncs; i++)
	{				
		perf = perf_test(userFuncs[i], funcNames[i], funcFlops[i]);
		printf("\nPerformance: %s\nPerf: %.3f FLOPs/c  Cycles: %.3f cycles\n", funcNames[i], perf, ((double)funcFlops[i])/perf);
	}


	return 0;
}




// Test for vanilla implementation
int validation(quant f){

	// Test input
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

	printf("Test Passed!!\n" );
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
double perf_test(quant f, char *desc, int flops)
{
	double cycles = 0.;
	double perf = 0.0;
	long num_runs = 100;
	double multiplier = 1;
	myInt64 start, end;

	double **d;
	uint8_t **q;
	double mn, mx;
	int n = 128;

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
	return (n * flops * 1.0) / cycles;
}

