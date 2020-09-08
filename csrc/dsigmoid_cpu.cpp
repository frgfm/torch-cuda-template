#include "dsigmoid.h"
#include <torch/types.h>
#include <ATen/CPUApplyUtils.h>

namespace cpu {

using at::CPU_tensor_apply2;

template <typename scalar_t>
void
d_sigmoid_kernel(
  torch::Tensor &output,
  const torch::Tensor &input
) {
  CPU_tensor_apply2<scalar_t,scalar_t>(
    output, input,
    [=] (scalar_t &out, const scalar_t &inp) {
      d_sigmoid_func(out, inp);
    }
  );
}

} // namespace cpu


void
d_sigmoid_cpu(
    torch::Tensor &output, const torch::Tensor &input
) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "d_sigmoid_cpu", [&] {
      cpu::d_sigmoid_kernel<scalar_t>(output, input);
  });
}
