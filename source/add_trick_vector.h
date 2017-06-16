#ifndef ADD_TRICK_VECTOR_H
#define ADD_TRICK_VECTOR_H

void add_trick_vector_naive(int16_t* acc, uint16_t* term2, uint16_t* term3, uint16_t term4, int n, int m);
void add_trick_vector_AVX(int16_t* acc, uint16_t* term2, uint16_t* term3, uint16_t term4, int n, int m);

#endif