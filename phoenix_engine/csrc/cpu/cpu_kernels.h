#pragma once

#include "core/tensor_data.h"

namespace phoenix {
namespace cpu {

// Basic element-wise addition
TensorDataPtr add(const TensorDataPtr& a, const TensorDataPtr& b);

// Basic element-wise multiplication
TensorDataPtr multiply(const TensorDataPtr& a, const TensorDataPtr& b);

// General Matrix Multiply (2D)
TensorDataPtr gemm(const TensorDataPtr& a, const TensorDataPtr& b);

} // namespace cpu
} // namespace phoenix
