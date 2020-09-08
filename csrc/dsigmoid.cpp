#include <torch/extension.h>
using namespace pybind11::literals;

// Forward declaration of kernels
void d_sigmoid_cuda(torch::Tensor &output, const torch::Tensor &input);

// CPU function
void d_sigmoid_cpu(torch::Tensor &output, const torch::Tensor &input) {
  auto s = torch::sigmoid(input);
  output = (1 - s) * s;
}

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Dispatch function
torch::Tensor
d_sigmoid(const torch::Tensor &input, const at::optional<torch::Tensor> out) {
  auto input_arg = torch::TensorArg(input, "input", 0);
  if (out) {
    auto out_arg = torch::TensorArg(*out, "out", 1);
    torch::checkSameType("d_sigmoid", input_arg, out_arg);
    torch::checkSameSize("d_sigmoid", input_arg, out_arg);
  }
  auto o = out.value_or(torch::empty_like(input));
  switch (input.device().type()) {
    case c10::kCUDA:
      CHECK_INPUT(input);
      d_sigmoid_cuda(o, input);
      break;
    case c10::kCPU:
      CHECK_CONTIGUOUS(input);
      d_sigmoid_cpu(o, input);
      break;
    default:
      TORCH_CHECK(false, "Unsupported device type, should be CUDA but got ", input.device().type());
  }
  return o;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("d_sigmoid", &d_sigmoid, "Sigmoid derivative", "input"_a, "out"_a = nullptr);
}