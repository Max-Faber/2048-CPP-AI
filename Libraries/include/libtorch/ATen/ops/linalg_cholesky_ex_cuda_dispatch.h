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

namespace cuda {

TORCH_API ::std::tuple<at::Tensor,at::Tensor> linalg_cholesky_ex(const at::Tensor & self, bool upper=false, bool check_errors=false);
TORCH_API ::std::tuple<at::Tensor &,at::Tensor &> linalg_cholesky_ex_out(at::Tensor & L, at::Tensor & info, const at::Tensor & self, bool upper=false, bool check_errors=false);
TORCH_API ::std::tuple<at::Tensor &,at::Tensor &> linalg_cholesky_ex_outf(const at::Tensor & self, bool upper, bool check_errors, at::Tensor & L, at::Tensor & info);

} // namespace cuda
} // namespace at
