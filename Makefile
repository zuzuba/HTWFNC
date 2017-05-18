# ==================================================================================== #
# = Executable
# ==================================================================================== #

BIN = program
TEST_BIN_QMM = test_qmm
TEST_BIN_QUANT = test_quant

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
OBJS = qmm.o quantize.o utils.o 

TEST_OBJ_QMM = test_qmm.o 
TEST_OBJ_QUANT = test_quantize.o
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

perf: $(OBJS) timing_qmm.o timing_quantize.o
	$(CC) $(OBJS) timing_qmm.o -o perf_qmm
	./perf_qmm
	$(CC) $(OBJS) timing_quantize.o -o perf_quantize
	./perf_quantize
	python performance_plot.py


%.o : %.cpp $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@



clean:
	rm -rf *.o *.txt *.s
	rm -rf $(BIN) $(TEST_BIN)