void qmmm( float lhs_scale, uint8_t lhs_offset, float rhs_scale,uint8_t rhs_offset, uint8_t* lhs_int_mat, uint8_t* rhs_int_mat, float result_scale, uint8_t result_offset, uint8_t* result_int_mat, int n,int k, int m ){

	int accumulator;

	for(int i=0;i<n;i++){
		for(int j=0;j<m;j++){
			for(int t=0; t<k;t++){
				accumulator = lhs_int_mat[i*n + t]*lhs_int_mat[t*m + j];
			}
		result_int_mat[i*n+j] = result_offset + (lhs_scale*rhs_scale/result_scale)*accumulator;
		}
	}
}