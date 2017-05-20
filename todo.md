# TO DO
* Quantized parameter returns zero point as float, it should be an integer
* Quantize_4x4 should return scale and offset
* Offset should be passed to qmm as int and not uint_4x4
* Implement a test function that compare the result of the optimized function versus the result of the vanilla implementation (that is tested already).

# Optimizations still need to be done
* [Done(Unroll)-Ladi] Unroll and scalar replacement for min max loop in quantization
* [Done-Ladi] Vectorization of min max loop in quantization
* [(semi)Done-Ladi] Vectorization of quantization (saturate would be nice to vectorizeb because it is used in qmm as well)
* [Ladi: Why do we need this? I thought we would just store the weights in transpose format and use dot-product] Blocking for qmm
* [Done-Ladi] In case it reuires a lot of time to execute, we might consider to a vectorized version of round (together with saturate this function is used in both qmm and quantization)
* Suggestions are welcome...
