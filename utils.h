float inline max(float a, float b);
float inline min(float a, float b);
int inline saturate(float a);
void quantize_parameter(float min, float max, float *scale, float *zero_point);
void get_min_max(float *d,int rows,int columns, float *min, float *max);