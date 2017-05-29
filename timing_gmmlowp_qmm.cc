// Example code illustrating the theory exposed in doc/quantization.md

/* Command line to build and run on x86:

c++ doc/quantization_example.cc -I . --std=c++11 -msse4.1 -lpthread \
  -o /tmp/quantization_example && \
/tmp/quantization_example

*/

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>
#include "gemmlowp/public/gemmlowp.h"
#include "gemmlowp/public/output_stages.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <list>
#include <time.h>
#include <sys/time.h>
#include "tsc_x86.h"



#define CYCLES_REQUIRED 1e7
#define REP 30
#define MAX_FUNCS 32
#define EPS (1e-3)

/* start ---definition relevant to timing */
#define NUM_RUNS 1        // number of runs we measure, then average
#define FREQUENCY 3.4e9     // need to change on different computers. This value is the actual CPU frequency
// #define CALIBRATE        // Whether calibrate before measuring.
/* end   ---definition relevant to timing */

using namespace std;
double perf_test(int dim);
#define TEST_FAILED -1


template <typename tScalar, gemmlowp::MapOrder tOrder>
std::ostream& operator<<(std::ostream& s,
                         const gemmlowp::MatrixMap<tScalar, tOrder>& m) {
  for (int i = 0; i < m.rows(); i++) {
    for (int j = 0; j < m.cols(); j++) {
      if (j) {
        s << '\t';
      }
      s << static_cast<float>(m(i, j));
    }
    s << '\n';
  }
  return s;
}

// Find the min and max value in a float matrix.
template <gemmlowp::MapOrder tOrder>
void FindMinMax(const gemmlowp::MatrixMap<float, tOrder>& m, float* min,
                float* max) {
  *min = *max = m(0, 0);
  for (int i = 0; i < m.rows(); i++) {
    for (int j = 0; j < m.cols(); j++) {
      const float val = m(i, j);
      *min = std::min(*min, val);
      *max = std::max(*max, val);
    }
  }
}


struct QuantizationParams {
  float scale;
  std::uint8_t zero_point;
};

// Given the min and max values of a float array, return
// reasonable quantization parameters to use for this array.
QuantizationParams ChooseQuantizationParams(float min, float max) {
  // representable value.
  min = std::min(min, 0.f);
  max = std::max(max, 0.f);

  // the min and max quantized values, as floating-point values
  const float qmin = 0;
  const float qmax = 255;

  // First determine the scale.
  const double scale = (max - min) / (qmax - qmin);

  const double initial_zero_point = qmin - min / scale;

 
  std::uint8_t nudged_zero_point = 0;
  if (initial_zero_point < qmin) {
    nudged_zero_point = qmin;
  } else if (initial_zero_point > qmax) {
    nudged_zero_point = qmax;
  } else {
    nudged_zero_point =
        static_cast<std::uint8_t>(std::round(initial_zero_point));
  }

  QuantizationParams result;
  result.scale = scale;
  result.zero_point = nudged_zero_point;
  return result;
}

template <gemmlowp::MapOrder tLhsOrder, gemmlowp::MapOrder tRhsOrder,
          gemmlowp::MapOrder tResultOrder>
void FloatMatrixMultiplication(
    const gemmlowp::MatrixMap<const float, tLhsOrder>& lhs,
    const gemmlowp::MatrixMap<const float, tRhsOrder>& rhs,
    gemmlowp::MatrixMap<float, tResultOrder>* result) {
  assert(lhs.cols() == rhs.rows());
  assert(lhs.rows() == result->rows());
  assert(rhs.cols() == result->cols());
  for (int i = 0; i < lhs.rows(); i++) {
    for (int k = 0; k < rhs.cols(); k++) {
      (*result)(i, k) = 0;
      for (int j = 0; j < lhs.cols(); j++) {
        (*result)(i, k) += lhs(i, j) * rhs(j, k);
      }
    }
  }
}

void Quantize(const QuantizationParams& qparams, const std::vector<float>& src,
              std::vector<std::uint8_t>* dst) {
  assert(src.size() == dst->size());
  for (std::size_t i = 0; i < src.size(); i++) {
    const float real_val = src[i];
    const float transformed_val = qparams.zero_point + real_val / qparams.scale;
    const float clamped_val = std::max(0.f, std::min(255.f, transformed_val));
    (*dst)[i] = static_cast<std::uint8_t>(std::round(clamped_val));
  }
}

void Dequantize(const QuantizationParams& qparams,
                const std::vector<std::uint8_t>& src, std::vector<float>* dst) {
  assert(src.size() == dst->size());
  for (std::size_t i = 0; i < src.size(); i++) {
    const std::uint8_t quantized_val = src[i];
    (*dst)[i] = qparams.scale * (quantized_val - qparams.zero_point);
  }
}

template <typename tScalar, gemmlowp::MapOrder tOrder>
class MatrixWithStorage {
 public:
  MatrixWithStorage(int rows, int cols)
      : storage(rows * cols), matrix_map(storage.data(), rows, cols) {}
  void MakeRandom() {
    static std::mt19937 random_engine;
    std::uniform_real_distribution<float> distribution(-1, 1);
    for (auto& x : storage) {
      x = static_cast<tScalar>(distribution(random_engine));
    }
  }
  gemmlowp::MatrixMap<const tScalar, tOrder> ConstMap() const {
    return gemmlowp::MatrixMap<const tScalar, tOrder>(
        storage.data(), matrix_map.rows(), matrix_map.cols());
  }
  gemmlowp::MatrixMap<tScalar, tOrder> Map() {
    return gemmlowp::MatrixMap<tScalar, tOrder>(
        storage.data(), matrix_map.rows(), matrix_map.cols());
  }
  const std::vector<tScalar>& Storage() const { return storage; }
  std::vector<tScalar>& Storage() { return storage; }

 private:
  std::vector<tScalar> storage;
  gemmlowp::MatrixMap<tScalar, tOrder> matrix_map;
};

template <typename tScalar, gemmlowp::MapOrder tOrder>
std::ostream& operator<<(std::ostream& s,
                         const MatrixWithStorage<tScalar, tOrder>& m) {
  return s << m.ConstMap();
}


void QuantizeMultiplierSmallerThanOne(float real_multiplier,
                                      std::int32_t* quantized_multiplier,
                                      int* right_shift) {
  assert(real_multiplier > 0.f);
  assert(real_multiplier < 1.f);
  int s = 0;
  // We want to bring the real multiplier into the interval [1/2, 1).
  // We can do so by multiplying it by two, and recording how many times
  // we multiplied by two so that we can compensate that by a right
  // shift by the same amount.
  while (real_multiplier < 0.5f) {
    real_multiplier *= 2.0f;
    s++;
  }
  // Now that the real multiplier is in [1/2, 1), we convert it
  // into a fixed-point number.
  std::int64_t q =
      static_cast<std::int64_t>(std::round(real_multiplier * (1ll << 31)));
  assert(q <= (1ll << 31));
  // Handle the special case when the real multiplier was so close to 1
  // that its fixed-point approximation was undistinguishable from 1.
  // We handle this by dividing it by two, and remembering to decrement
  // the right shift amount.
  if (q == (1ll << 31)) {
    q /= 2;
    s--;
  }
  assert(s >= 0);
  assert(q <= std::numeric_limits<std::int32_t>::max());
  *quantized_multiplier = static_cast<std::int32_t>(q);
  *right_shift = s;
}


int main(int argc, char **argv)
//int test_qmm()
{
  printf("------Timing function google qmm-----\n");
  float cycles,perf;
    // Initialize the vectors of functions, function names and function flops
  // Test of vanilla implementation first

  printf("Performance of gemmlowp qmm function:\n");
  FILE *fp = fopen("data/perf__gemm_qmm.dat","w+");
  FILE *fp_cycles = fopen("data/cycles__gemm_qmm.dat","w+");

  for(int n=30; n<500;n+=30){
    cycles = perf_test(n);
    perf = 2*n*n*n/cycles;
    printf("gemmlowp: n:%d cycles:%f perf:%f \n", n, cycles,perf);
    fprintf(fp, "%d %f\n",n,perf);
    fprintf(fp_cycles, "%d %f\n",n,cycles);
  } 


  return 0;
}

double perf_test(int n)
{
  double cycles = 0.;
  double perf = 0.0;
  long num_runs = 100;
  double multiplier = 1;
  myInt64 start, end;

  const auto kOrder = gemmlowp::MapOrder::ColMajor;
  MatrixWithStorage<float, kOrder> float_lhs(n, n);
  float_lhs.MakeRandom();
  MatrixWithStorage<float, kOrder> float_rhs(n, n);
  float_rhs.MakeRandom();
  MatrixWithStorage<float, kOrder> reference_float_result(n, n);

  auto reference_float_result_map = reference_float_result.Map();
  FloatMatrixMultiplication(float_lhs.ConstMap(), float_rhs.ConstMap(),
                            &reference_float_result_map);


  float lhs_min, lhs_max, rhs_min, rhs_max, result_min, result_max;
  FindMinMax(float_lhs.Map(), &lhs_min, &lhs_max);
  FindMinMax(float_rhs.Map(), &rhs_min, &rhs_max);
  FindMinMax(reference_float_result.Map(), &result_min, &result_max);
  const auto lhs_qparams = ChooseQuantizationParams(lhs_min, lhs_max);
  const auto rhs_qparams = ChooseQuantizationParams(rhs_min, rhs_max);
  const auto result_qparams = ChooseQuantizationParams(result_min, result_max);


  MatrixWithStorage<std::uint8_t, kOrder> uint8_lhs(n, n);
  MatrixWithStorage<std::uint8_t, kOrder> uint8_rhs(n, n);
  MatrixWithStorage<std::uint8_t, kOrder> actual_uint8_result(n, n);

  Quantize(lhs_qparams, float_lhs.Storage(), &uint8_lhs.Storage());
  Quantize(rhs_qparams, float_rhs.Storage(), &uint8_rhs.Storage());

  // std::cout << "Quantized uint8 LHS matrix:\n" << uint8_lhs << std::endl;
  // std::cout << "Quantized uint8 RHS matrix:\n" << uint8_rhs << std::endl;

  const int lhs_offset = -lhs_qparams.zero_point;
  const int rhs_offset = -rhs_qparams.zero_point;
  const int result_offset = result_qparams.zero_point;

  const float real_multiplier =
      lhs_qparams.scale * rhs_qparams.scale / result_qparams.scale;
  std::int32_t quantized_multiplier;
  int right_shift;
  QuantizeMultiplierSmallerThanOne(real_multiplier, &quantized_multiplier,
                                   &right_shift);


  gemmlowp::OutputStageQuantizeDownInt32ToUint8ScaleByFixedPoint
      quantize_down_stage;
  quantize_down_stage.result_offset_after_shift = result_offset;
  quantize_down_stage.result_fixedpoint_multiplier = quantized_multiplier;
  quantize_down_stage.result_shift = right_shift;
  gemmlowp::OutputStageSaturatingCastToUint8 saturating_cast_stage;
  const auto& output_pipeline =
      std::make_tuple(quantize_down_stage, saturating_cast_stage);

  

  // Warm-up phase: we determine a number of executions that allows
  // the code to be executed for at least CYCLES_REQUIRED cycles.
  // This helps excluding timing overhead when measuring small runtimes.
  do {
    num_runs = num_runs * multiplier;
    start = start_tsc();
    for (size_t i = 0; i < num_runs; i++) {
      auto actual_uint8_result_map = actual_uint8_result.Map();
      gemmlowp::GemmContext gemm_context;
      gemmlowp::GemmWithOutputPipeline<std::uint8_t, std::uint8_t,
                                   gemmlowp::DefaultL8R8BitDepthParams>(
      &gemm_context, uint8_lhs.ConstMap(), uint8_rhs.ConstMap(),
      &actual_uint8_result_map, lhs_offset, rhs_offset, output_pipeline);  
    }
    end = stop_tsc(start);

    cycles = (double)end;
    multiplier = (CYCLES_REQUIRED) / (cycles);
    
  } while (multiplier > 2);

  list< double > cyclesList;

  // Actual performance measurements repeated REP times.
  // We simply store all results and compute medians during post-processing.
  for (size_t j = 0; j < REP; j++) {

    start = start_tsc();
    for (size_t i = 0; i < num_runs; ++i) {
      auto actual_uint8_result_map = actual_uint8_result.Map();
      gemmlowp::GemmContext gemm_context;
      gemmlowp::GemmWithOutputPipeline<std::uint8_t, std::uint8_t,
                                   gemmlowp::DefaultL8R8BitDepthParams>(
      &gemm_context, uint8_lhs.ConstMap(), uint8_rhs.ConstMap(),
      &actual_uint8_result_map, lhs_offset, rhs_offset, output_pipeline);
    }
    end = stop_tsc(start);

    cycles = ((double)end) / num_runs;

    cyclesList.push_back(cycles);
  }

  cyclesList.sort();
  cycles = cyclesList.front();

  return cycles;
}