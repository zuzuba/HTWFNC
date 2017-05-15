# ==================================================================================== #
# = Executable
# ==================================================================================== #

BIN = program
TEST_BIN = perftest

# ==================================================================================== #
# = Compiler settings
# ==================================================================================== #

#CC      = gcc
#CFLAGS += -O3 -no-vec

CC      = g++
CFLAGS += -O3 -fno-tree-vectorize -march=native

# ==================================================================================== #
# = Object Files
# ==================================================================================== #

# %.o : %.cpp
# 	$(CC) $(CFLAGS) -c $< -o $@

# %.s : %.cpp
#	$(CC) $(CFLAGS) -S $< -o $@


HEADERS=$(wildcard *.h)
OBJS = naive_qmm.o naive_quantize.o utils.o 
TEST_OBJ = naive_qmm_test.o main_tests.cpp
MAIN_OBJ = main.o

all: $(OBJS) 
	$(CC) $(OBJS) -o $(BIN)

main : $(OBJS) $(MAIN_OBJ)
	$(CC) $(OBJS) $(MAIN_OBJ) -o $(BIN)

test: $(OBJS) $(TEST_OBJ)
	$(CC) $(OBJS) $(TEST_OBJ) -o $(TEST_BIN)
	./$(TEST_BIN)

%.o : %.cpp $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@



clean:
	rm -rf *.o *.txt *.s
	rm -rf $(BIN) $(TEST_BIN)
