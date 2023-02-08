#pragma once
// @generated by torchgen/gen.py from DispatchKeyFunction.h

// NB: The implementing C++ file is RegisterDispatchKey.cpp

// The only #includes we need are for custom classes that have defaults in the C++ API
#include <c10/core/MemoryFormat.h>
#include <c10/core/Scalar.h>
#include <ATen/core/Reduction.h>

// Forward declarations of any types needed in the operator signatures.
// We can't directly include these classes because it will cause circular include dependencies.
// This file is included by TensorBody.h, which defines the Tensor class.
#include <ATen/core/ATen_fwd.h>

namespace at {

namespace cpu {

TORCH_API at::Tensor _softmax(const at::Tensor & self, int64_t dim, bool half_to_float);
TORCH_API at::Tensor & _softmax_out(at::Tensor & out, const at::Tensor & self, int64_t dim, bool half_to_float);
TORCH_API at::Tensor & _softmax_outf(const at::Tensor & self, int64_t dim, bool half_to_float, at::Tensor & out);

} // namespace cpu
} // namespace at
