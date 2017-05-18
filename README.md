# HTWFNC
Project for the course "How to write fast numerical code" 

## Adding a new function
If you want to add a new version of the quantization or the qmm just open the [quantize.cpp](./quantize.cpp) or [qmm.cpp](qmm.cpp) respectively. Add your new definitiion there following the signature that the other functions have. Add your deckaration to the header file.

## Testing a new function
After you have added a new function, you can test it by adding it in the register_functions() of [test_quantize.cpp](test_quantize.cpp) or [test_qmm.cpp](test_qmm.cpp), saving and typing make test in the terminal.

## Timing a new function
After you have added a new function, you can time it by adding it in the register_functions() of [timing_quantize.cpp](timing_quantize.cpp) or [timing_qmm.cpp](timing_qmm.cpp), saving and typing make perf in the terminal. This should produce automatic performance plots.


[Performance plot of for the naive implementation](plots/Performance_q.png)
