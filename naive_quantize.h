#ifndef NAIV_QUANTIZE_H
#define NAIV_QUANTIZE_H

void vanilla_quantize(float *d, uint4x1_t *q, float *min, float *max, int rows, int columns);

#endif
