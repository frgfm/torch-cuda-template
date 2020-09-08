#include <torch/types.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <c10/util/Half.h>
#define GLOBAL_INLINE __forceinline__ __host__ __device__
#else
#include <cmath>
#define GLOBAL_INLINE __inline__
#endif

template <typename scalar_t>
GLOBAL_INLINE
void d_sigmoid_func(scalar_t &out, const scalar_t &inp) {
  auto s = scalar_t(1.0) / (scalar_t(1.0) + exp(-inp));
  out = (1 - s) * s;
};

// Cast FP16 to FP32
template <>
GLOBAL_INLINE
void d_sigmoid_func(c10::Half &out, const c10::Half &inp) {
  float res;
  d_sigmoid_func<float>(res, (float)inp);
  out = res;
};
