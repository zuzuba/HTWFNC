# ==================================================================================== #
# = Executable
# ==================================================================================== #

BIN = program
TEST_BIN_QMM = test_qmm
TEST_BIN_QUANT = test_quant
TEST_BIN_ALL = validation
# ==================================================================================== #
# = Compiler settings
# ==================================================================================== #

#CC      = gcc
#CFLAGS += -O3 -no-vec

CC      = g++
CFLAGS += -O3 -fno-tree-vectorize -march=native -mavx


# ==================================================================================== #
# = Object Files
# ==================================================================================== #

# %.o : %.cpp
# 	$(CC) $(CFLAGS) -c $< -o $@

# %.s : %.cpp
#	$(CC) $(CFLAGS) -S $< -o $@


HEADERS=$(wildcard *.h)
OBJS = qmm.o quantize.o utils.o trick_vector.o qmm_kernel.o round_saturation.o add_trick_vector.o

TEST_OBJ_QMM = test_qmm.o 
TEST_OBJ_QUANT = test_quantize.o
TEST_OBJ_ALL = validation.o
MAIN_OBJ = main.o

all: $(OBJS) 
	$(CC) $(OBJS) -o $(BIN)

main : $(OBJS) $(MAIN_OBJ)
	$(CC) $(OBJS) $(MAIN_OBJ) -o $(BIN)

test: $(OBJS) $(TEST_OBJ_QMM) $(TEST_OBJ_QUANT)
	$(CC) $(OBJS) $(TEST_OBJ_QMM) -o $(TEST_BIN_QMM)
	./$(TEST_BIN_QMM)
	$(CC) $(OBJS) $(TEST_OBJ_QUANT) -o $(TEST_BIN_QUANT)
	./$(TEST_BIN_QUANT)

validation: $(OBJS) $(TEST_OBJ_ALL)
	$(CC) $(OBJS) $(TEST_OBJ_ALL) -o $(TEST_BIN_ALL)
	./$(TEST_BIN_ALL)

perf: $(OBJS) timing_qmm.o timing_quantize.o timing_add_trick_vector.o timing_trick_vector.o timing_round_saturation.o timing_qmm_kernel.o
	$(CC) $(OBJS) timing_qmm.o -o perf_qmm
	./perf_qmm
	$(CC) $(OBJS) timing_quantize.o -o perf_quantize
	./perf_quantize
	$(CC) $(OBJS) timing_qmm_kernel.o -o perf_qmm_kernel
	./perf_qmm_kernel
	$(CC) $(OBJS) timing_add_trick_vector.o -o perf_add_trick_vector
	./perf_add_trick_vector
	$(CC) $(OBJS) timing_trick_vector.o -o perf_trick_vector
	./perf_trick_vector
	$(CC) $(OBJS) timing_round_saturation.o -o perf_round_saturation
	./perf_round_saturation
	python performance_plot.py

gemm_quant: 
	clang++ -std=c++11 -stdlib=libc++ -O3 -fno-tree-vectorize -march=native -mavx timing_gmmlowp_quantize.cc -o timing_gmmlowp_quantize
	./timing_gmmlowp_quantize
gemm_qmm:
	clang++ -std=c++11 -stdlib=libc++ -O3 -fno-tree-vectorize -march=native -mavx timing_gmmlowp_qmm.cc -o timing_gmmlowp_qmm
	./timing_gmmlowp_qmm

%.o : %.cpp $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@



clean:
	rm -rf *.o *.txt *.s
	rm -rf $(BIN) $(TEST_BIN)