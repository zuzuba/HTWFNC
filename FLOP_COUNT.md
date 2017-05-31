# Flop Count

*quantize: NxM (1 add,1 div, 4 comparison,1 round): 7 NxM FLOPS

*trick vector: 2 NxN + 2 N integer operations

*qmm_kernel: 2 NxNxN integer operations

*add_trick_vecotr: 3 NxN integer ops

*round_Saturate: 5 NxN flops
