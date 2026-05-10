#pragma once

#include "core/tensor_data.h"

namespace phoenix {
namespace cpu {

// Basic element-wise addition
TensorDataPtr add(const TensorDataPtr& a, const TensorDataPtr& b);

// Basic element-wise subtraction
TensorDataPtr sub(const TensorDataPtr& a, const TensorDataPtr& b);

// Basic element-wise multiplication
TensorDataPtr multiply(const TensorDataPtr& a, const TensorDataPtr& b);

// General Matrix Multiply (2D)
TensorDataPtr gemm(const TensorDataPtr& a, const TensorDataPtr& b);

// Sum of all elements (Reduction)
TensorDataPtr sum(const TensorDataPtr& a);

// ReLU Activation
TensorDataPtr relu(const TensorDataPtr& a);

// Random Initialization (Standard Normal)
TensorDataPtr randn(const std::vector<size_t>& shape, DType dtype);

// Fused Softmax (along the last dimension)
TensorDataPtr softmax(const TensorDataPtr& a);

// Fused LayerNorm (along the last dimension)
TensorDataPtr layernorm(const TensorDataPtr& a, const TensorDataPtr& weight, const TensorDataPtr& bias, float eps);

// Embedding lookup
TensorDataPtr embedding(const TensorDataPtr& weight, const TensorDataPtr& indices);

} // namespace cpu
} // namespace phoenix
