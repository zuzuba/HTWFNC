	__m256i tmp = _mm256_loadu_si256((__m256i const *)a);
	
	// Mask for odd indices in memory
	__m256i odd = _mm256_and_si256(tmp,_mm256_set1_epi8(15));
	
	// Mask for even indices in memory
	__m256i even = _mm256_and_si256(tmp,_mm256_set1_epi8(240));
	
	// Shift and blend to recover rows
	b_mask = _mm256_set1_epi16(32768);
	*b1 = _mm256_blendv_epi8(odd,_mm256_slli_epi64(even, 4),b_mask);
	*b2 = _mm256_blendv_epi8(_mm256_srli_epi64(odd, 8),_mm256_srli_epi64(even, 4),b_mask);

