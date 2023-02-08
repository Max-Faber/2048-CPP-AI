#pragma once

// @generated by torchgen/gen.py from Function.h

#include <ATen/Context.h>
#include <ATen/DeviceGuard.h>
#include <ATen/TensorUtils.h>
#include <ATen/TracerMode.h>
#include <ATen/core/Generator.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>



#include <ATen/ops/_fake_quantize_learnable_per_channel_affine_ops.h>

namespace at {


// aten::_fake_quantize_learnable_per_channel_affine(Tensor self, Tensor scale, Tensor zero_point, int axis, int quant_min, int quant_max, float grad_factor=1.0) -> Tensor
inline at::Tensor _fake_quantize_learnable_per_channel_affine(const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max, double grad_factor=1.0) {
    return at::_ops::_fake_quantize_learnable_per_channel_affine::call(self, scale, zero_point, axis, quant_min, quant_max, grad_factor);
}

// aten::_fake_quantize_learnable_per_channel_affine.out(Tensor self, Tensor scale, Tensor zero_point, int axis, int quant_min, int quant_max, float grad_factor=1.0, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & _fake_quantize_learnable_per_channel_affine_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max, double grad_factor=1.0) {
    return at::_ops::_fake_quantize_learnable_per_channel_affine_out::call(self, scale, zero_point, axis, quant_min, quant_max, grad_factor, out);
}
// aten::_fake_quantize_learnable_per_channel_affine.out(Tensor self, Tensor scale, Tensor zero_point, int axis, int quant_min, int quant_max, float grad_factor=1.0, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & _fake_quantize_learnable_per_channel_affine_outf(const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max, double grad_factor, at::Tensor & out) {
    return at::_ops::_fake_quantize_learnable_per_channel_affine_out::call(self, scale, zero_point, axis, quant_min, quant_max, grad_factor, out);
}

}
