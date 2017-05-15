#include <stdio.h>
#include "function_tests.h"

/* 
To add a test declare it in function_test.h, write a separate cpp file to perform the tests and call the \
function to perform the test in the  main
*/
int main(){
	int test_status = 0;
	
	printf("Starting qmmm test\n");
	test_status += test_qmm();
}