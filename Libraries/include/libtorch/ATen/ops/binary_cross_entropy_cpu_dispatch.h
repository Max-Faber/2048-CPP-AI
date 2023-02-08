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

TORCH_API at::Tensor binary_cross_entropy(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight={}, int64_t reduction=at::Reduction::Mean);
TORCH_API at::Tensor & binary_cross_entropy_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight={}, int64_t reduction=at::Reduction::Mean);
TORCH_API at::Tensor & binary_cross_entropy_outf(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, at::Tensor & out);

} // namespace cpu
} // namespace at
