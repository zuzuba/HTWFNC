# TO DO
* Quantized parameter returns zero point as float, it should be an integer
* Quantize_4x4 should return scale and offset
* Offset should be passed to qmm as int and not uint_4x4

##IMPORTANT
*In main the result of qmm is always equal to result offset for all the classes. Why??