#define qmin = 0
#define qmax = 15

float inline max(float a, float b){
	return a < b ? b : a;
}

float inline min(float a, float b){
	return a < b ? a : b;
}

int inline saturate(float a){
	return (int)max(qmin,min(qmax,a));
}

void quantize_parameter(float min, float max, float *scale, float *zero_point){
	*scale = (max - min)/(qmax-qmin);
	*zero_point = -min/(*scale);
}

void get_min_max(float *d,int rows,int columns, float *min, float *max){
	*max = d[0];
	*min = *max;
	for(int i = 0; i<rows; i++){
		for(int j = 0; j<columns; j++){
			el = d[i*rows+columns];
			*max = max(mx,el);
			*min = min(mn,el);
		}
	}
}
