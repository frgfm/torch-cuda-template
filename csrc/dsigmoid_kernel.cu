#include <torch/types.h>
#include <cuda_runtime.h>
// cf. https://discuss.pytorch.org/t/error-when-building-an-extension/63317/7
// cf. https://discuss.pytorch.org/t/cuda-tensor-apply-in-extension-gives-undefined-symbol/56736/4
// #include <ATen/cuda/CUDAApplyUtils.cuh>
#include "CUDAApplyUtils.cuh"

// TORCH_CHECK replaces AT_CHECK in PyTorch 1,2, support 1.1 as well.
#ifndef TORCH_CHECK
#define TORCH_CHECK AT_CHECK
#endif

#ifndef __CUDACC_EXTENDED_LAMBDA__
#error "please compile with --expt-extended-lambda"
#endif

namespace kernel {
#include "dsigmoid.h"

using at::cuda::CUDA_tensor_apply2;
using at::cuda::TensorArgType;

template <typename scalar_t>
void
d_sigmoid_kernel(
  torch::Tensor &output,
  const torch::Tensor &input
) {
  CUDA_tensor_apply2<scalar_t,scalar_t>(
    output, input,
    [=] __host__ __device__ (scalar_t &out, const scalar_t &inp) {
      d_sigmoid_func(out, inp);
    },
    TensorArgType::ReadWrite, TensorArgType::ReadOnly
  );
}

} // namespace kernel

void
d_sigmoid_cuda(
    torch::Tensor &output, const torch::Tensor &input
) {
  auto in_arg  = torch::TensorArg(input,  "input",  0),
       out_arg = torch::TensorArg(output, "output", 1);
  torch::checkAllDefined("d_sigmoid_cuda", {in_arg, out_arg});
  torch::checkAllSameGPU("d_sigmoid_cuda", {in_arg, out_arg});
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "d_sigmoid_cuda", [&] {
      kernel::d_sigmoid_kernel<scalar_t>(output, input);
  });
}
