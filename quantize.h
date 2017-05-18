#ifndef QUANTIZE_H
#define QUANTIZE_H

void vanilla_quantize(float *d, uint4x1_t *q, float *min, float *max, int rows, int columns);
void quantize_4x4(float *d, uint4x4_t *q, float *min, float *max, int rows, int columns);

#endif
