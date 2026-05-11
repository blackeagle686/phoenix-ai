#pragma once

#include <vector>
#include <memory>
#include "types.h"

namespace phoenix {

// Forward declaration
class TensorData;
using TensorDataPtr = std::shared_ptr<TensorData>;

class Backend {
public:
    virtual ~Backend() = default;

    // Memory Management
    virtual void* allocate(size_t num_bytes) = 0;
    virtual void deallocate(void* ptr) = 0;

    // Basic Ops
    virtual TensorDataPtr add(const TensorDataPtr& a, const TensorDataPtr& b) = 0;
    virtual TensorDataPtr add_scalar(const TensorDataPtr& a, float scalar) = 0;
    virtual TensorDataPtr sub(const TensorDataPtr& a, const TensorDataPtr& b) = 0;
    virtual TensorDataPtr multiply(const TensorDataPtr& a, const TensorDataPtr& b) = 0;
    virtual TensorDataPtr multiply_scalar(const TensorDataPtr& a, float scalar) = 0;
    virtual TensorDataPtr divide(const TensorDataPtr& a, const TensorDataPtr& b) = 0;
    virtual TensorDataPtr divide_scalar(const TensorDataPtr& a, float scalar) = 0;

    // Activation & Normalization
    virtual TensorDataPtr relu(const TensorDataPtr& a) = 0;
    virtual TensorDataPtr softmax(const TensorDataPtr& a) = 0;
    virtual TensorDataPtr layernorm(const TensorDataPtr& a, const TensorDataPtr& weight, const TensorDataPtr& bias, float eps) = 0;

    // Linear Algebra
    virtual TensorDataPtr gemm(const TensorDataPtr& a, const TensorDataPtr& b) = 0;

    // Specialized Ops
    virtual TensorDataPtr embedding_forward(const TensorDataPtr& indices, const TensorDataPtr& weight) = 0;
    virtual TensorDataPtr masked_fill(const TensorDataPtr& a, const TensorDataPtr& mask, float value) = 0;
    virtual TensorDataPtr tril(const std::vector<size_t>& shape) = 0;
    virtual TensorDataPtr narrow(const TensorDataPtr& a, size_t dim, size_t start, size_t length) = 0;

    // Initializers
    virtual TensorDataPtr randn(const std::vector<size_t>& shape, DType dtype) = 0;
    virtual TensorDataPtr zeros(const std::vector<size_t>& shape, DType dtype) = 0;
    virtual TensorDataPtr ones(const std::vector<size_t>& shape, DType dtype) = 0;

    // Loss functions
    virtual TensorDataPtr softmax_cross_entropy(const TensorDataPtr& logits, const TensorDataPtr& targets) = 0;
    virtual TensorDataPtr softmax_cross_entropy_backward(const TensorDataPtr& grad_out, const TensorDataPtr& logits, const TensorDataPtr& targets) = 0;
};

} // namespace phoenix
