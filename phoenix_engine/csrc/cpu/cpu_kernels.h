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

} // namespace cpu
} // namespace phoenix
