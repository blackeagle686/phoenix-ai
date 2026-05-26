#pragma once

#include "../core/backend.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace phoenix {

class CUDABackend : public Backend {
public:
    CUDABackend() = default;
    ~CUDABackend() = default;

    // Memory Management
    void* allocate(size_t num_bytes) override;
    void deallocate(void* ptr) override;

    // Basic Ops
    TensorDataPtr add(const TensorDataPtr& a, const TensorDataPtr& b) override;
    TensorDataPtr add_scalar(const TensorDataPtr& a, float scalar) override;
    TensorDataPtr sub(const TensorDataPtr& a, const TensorDataPtr& b) override;
    TensorDataPtr multiply(const TensorDataPtr& a, const TensorDataPtr& b) override;
    TensorDataPtr multiply_scalar(const TensorDataPtr& a, float scalar) override;
    TensorDataPtr divide(const TensorDataPtr& a, const TensorDataPtr& b) override;
    TensorDataPtr divide_scalar(const TensorDataPtr& a, float scalar) override;
    TensorDataPtr sqrt(const TensorDataPtr& a) override;
    TensorDataPtr sum(const TensorDataPtr& a) override;

    // Activation & Normalization 
    TensorDataPtr relu(const TensorDataPtr& a) override;
    TensorDataPtr softmax(const TensorDataPtr& a) override;
    TensorDataPtr layernorm(const TensorDataPtr& a, const TensorDataPtr& weight, const TensorDataPtr& bias, float eps) override;

    // Linear Algebra
    TensorDataPtr gemm(const TensorDataPtr& a, const TensorDataPtr& b) override;

    // Specialized Ops
    TensorDataPtr embedding_forward(const TensorDataPtr& indices, const TensorDataPtr& weight) override;
    TensorDataPtr masked_fill(const TensorDataPtr& a, const TensorDataPtr& mask, float value) override;
    TensorDataPtr tril(const std::vector<size_t>& shape) override;
    TensorDataPtr narrow(const TensorDataPtr& a, size_t dim, size_t start, size_t length) override;

    // Initializers
    TensorDataPtr randn(const std::vector<size_t>& shape, DType dtype) override;
    TensorDataPtr zeros(const std::vector<size_t>& shape, DType dtype) override;
    TensorDataPtr ones(const std::vector<size_t>& shape, DType dtype) override;

    // Loss functions
    TensorDataPtr softmax_cross_entropy(const TensorDataPtr& logits, const TensorDataPtr& targets) override;
    TensorDataPtr softmax_cross_entropy_backward(const TensorDataPtr& grad_out, const TensorDataPtr& logits, const TensorDataPtr& targets) override;
};

} // namespace phoenix
