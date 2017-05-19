# TO DO
* Quantized parameter returns zero point as float, it should be an integer
* Quantize_4x4 should return scale and offset
* Offset should be passed to qmm as int and not uint_4x4
* Implement a test function that compare the result of the optimized function versus the result of the vanilla implementation (that is tested already).