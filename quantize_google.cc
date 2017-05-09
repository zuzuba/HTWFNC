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
#include "../gemmlowp/public/gemmlowp.h"
#include "../gemmlowp/public/output_stages.h"
#include <time.h>
#include <sys/time.h>
#include "tsc_x86.h"
#include <list>

#define CYCLES_REQUIRED 1e7
#define REP 30


// We will handle both float and quantized matrices, which we will
// represent as gemmlowp::MatrixMap.
// We will need to be able to print them.

// Output a matrix to a std::ostream
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

// A structure to hold quantization parameters 'scale' and 'zero_point'
// as discussed in doc/quantization.md. As explained there, the meaning
// of these values is as the constants in the quantization equation
//
//   real_value = scale * (quantized_value - zero_point)
//
// In other words, 'zero_point' is the quantized value that corresponds
// to the real value 0, and 'scale' is the difference of real values
// corresponding to consecutive quantized values.
struct QuantizationParams {
  float scale;
  std::uint8_t zero_point;
};

// Given the min and max values of a float array, return
// reasonable quantization parameters to use for this array.
QuantizationParams ChooseQuantizationParams(float min, float max) {
  // We extend the [min, max] interval to ensure that it contains 0.
  // Otherwise, we would not meet the requirement that 0 be an exactly
  // representable value.
  min = std::min(min, 0.f);
  max = std::max(max, 0.f);

  // the min and max quantized values, as floating-point values
  const float qmin = 0;
  const float qmax = 255;

  // First determine the scale.
  const double scale = (max - min) / (qmax - qmin);

  // Zero-point computation.
  // First the initial floating-point computation. The zero-point can be
  // determined from solving an affine equation for any known pair
  // (real value, corresponding quantized value).
  // We know two such pairs: (rmin, qmin) and (rmax, qmax).
  // Let's use the first one here.
  const double initial_zero_point = qmin - min / scale;

  // Now we need to nudge the zero point to be an integer
  // (our zero points are integer, and this is motivated by the requirement
  // to be able to represent the real value "0" exactly as a quantized value,
  // which is required in multiple places, for example in Im2col with SAME
  // padding).
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

// Given a real_multiplier in the interval (0, 1),
// produces a pair (quantized_multiplier, right_shift) where
// quantized_multiplier is an int32 representing a fixed-point value
// in the interval [-1, 1)  (in practice we only produce positive values)
// and right_shift is an amount to shift right by, so that the
// floating-point multiplication of some int32 input value by real_multiplier,
//
//   return static_cast<int32>(int32_value * real_multiplier);
//
// is best approximated by the integer-arithmetic-only code
//
//   return RoundingRightShift(
//       FixedPointMultiplication(int32_value, quantized_multiplier),
//       right_shift);
//
// This is how to obtain the fixed-point multiplier and right shift
// parameters to pass to
// OutputStageQuantizeDownInt32ToUint8ScaleByFixedPoint.
//
// Note: all this code only needs to run offline to generate the quantized
// neural network workload, not at runtime on the
// device on which quantized neural networks need to run. So it's not
// performance-critical at all.
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

template <gemmlowp::MapOrder tOrder>
void QuantizeGoogle(const gemmlowp::MatrixMap<float, tOrder>& m, const std::vector<float>& src, std::vector<std::uint8_t>* dst){

  float mn, mx;
  FindMinMax(m, &mn, &mx);
  const auto qparams = ChooseQuantizationParams(mn,mx);
  Quantize(qparams, src, dst);

}

int main() {

  double cycles = 0.;
  double perf = 0.0;
  long num_runs = 100;
  double multiplier = 1;
  double flops = 7.;
  myInt64 start, end;
  FILE *fp = fopen("quantize_google.dat","w+");

  for(int n= 3;n<=200;n+=1){
    
    printf("%d: ",n);
    int rows = n;
    int depth = n;
    int cols = n;
    const auto kOrder = gemmlowp::MapOrder::ColMajor;

    MatrixWithStorage<float, kOrder> float_mat(rows, depth);
    float_mat.MakeRandom();

    float mn, mx;
    MatrixWithStorage<std::uint8_t, kOrder> uint8_mat(rows, depth);

    const gemmlowp::MatrixMap<float, kOrder>& m = float_mat.Map();
    const std::vector<float>& src = float_mat.Storage();
    std::vector<std::uint8_t>* dst = &uint8_mat.Storage();

    // Warm-up phase: we determine a number of executions that allows
    // the code to be executed for at least CYCLES_REQUIRED cycles.
    // This helps excluding timing overhead when measuring small runtimes.
    do {
      num_runs = num_runs * multiplier;
      start = start_tsc();
      for (size_t i = 0; i < num_runs; i++) {
        QuantizeGoogle(m,src,dst);     
      }
      end = stop_tsc(start);

      cycles = (double)end;
      multiplier = (CYCLES_REQUIRED) / (cycles);
      
    } while (multiplier > 2);

    std::list< double > cyclesList;

    // Actual performance measurements repeated REP times.
    // We simply store all results and compute medians during post-processing.
    for (size_t j = 0; j < REP; j++) {

      start = start_tsc();
      for (size_t i = 0; i < num_runs; ++i) {
        QuantizeGoogle(m,src,dst);   
      }
      end = stop_tsc(start);

      cycles = ((double)end) / num_runs;

      cyclesList.push_back(cycles);
  
    }

    cyclesList.sort();
    cycles = cyclesList.front();

    perf =  (n*n*flops*1.0) / cycles;

    printf("%f \n", perf);
    fprintf(fp, "%d %f\n",n,perf);
  }
}