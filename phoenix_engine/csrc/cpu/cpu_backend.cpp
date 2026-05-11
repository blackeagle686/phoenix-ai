#include "cpu_backend.h"
#include "cpu_kernels.h"
#include <cstdlib>

namespace phoenix {

void* CPUBackend::allocate(size_t num_bytes) {
    void* ptr = std::malloc(num_bytes);
    if (!ptr) throw std::runtime_error("CPUBackend: Failed to allocate " + std::to_string(num_bytes) + " bytes.");
    return ptr;
}

void CPUBackend::deallocate(void* ptr) {
    std::free(ptr);
}

TensorDataPtr CPUBackend::add(const TensorDataPtr& a, const TensorDataPtr& b) {
    return cpu::add(a, b);
}

TensorDataPtr CPUBackend::add_scalar(const TensorDataPtr& a, float scalar) {
    return cpu::add_scalar(a, scalar);
}

TensorDataPtr CPUBackend::sub(const TensorDataPtr& a, const TensorDataPtr& b) {
    return cpu::sub(a, b);
}

TensorDataPtr CPUBackend::multiply(const TensorDataPtr& a, const TensorDataPtr& b) {
    return cpu::multiply(a, b);
}

TensorDataPtr CPUBackend::multiply_scalar(const TensorDataPtr& a, float scalar) {
    return cpu::multiply_scalar(a, scalar);
}

TensorDataPtr CPUBackend::divide(const TensorDataPtr& a, const TensorDataPtr& b) {
    return cpu::divide(a, b);
}

TensorDataPtr CPUBackend::divide_scalar(const TensorDataPtr& a, float scalar) {
    return cpu::divide_scalar(a, scalar);
}

TensorDataPtr CPUBackend::sqrt(const TensorDataPtr& a) {
    return cpu::sqrt(a);
}

TensorDataPtr CPUBackend::sum(const TensorDataPtr& a) {
    return cpu::sum(a);
}

TensorDataPtr CPUBackend::relu(const TensorDataPtr& a) {
    return cpu::relu(a);
}

TensorDataPtr CPUBackend::softmax(const TensorDataPtr& a) {
    return cpu::softmax(a);
}

TensorDataPtr CPUBackend::layernorm(const TensorDataPtr& a, const TensorDataPtr& weight, const TensorDataPtr& bias, float eps) {
    return cpu::layernorm(a, weight, bias, eps);
}

TensorDataPtr CPUBackend::gemm(const TensorDataPtr& a, const TensorDataPtr& b) {
    return cpu::gemm(a, b);
}

TensorDataPtr CPUBackend::embedding_forward(const TensorDataPtr& indices, const TensorDataPtr& weight) {
    return cpu::embedding_forward(indices, weight);
}

TensorDataPtr CPUBackend::masked_fill(const TensorDataPtr& a, const TensorDataPtr& mask, float value) {
    return cpu::masked_fill(a, mask, value);
}

TensorDataPtr CPUBackend::tril(const std::vector<size_t>& shape) {
    return cpu::tril(shape);
}

TensorDataPtr CPUBackend::narrow(const TensorDataPtr& a, size_t dim, size_t start, size_t length) {
    return cpu::narrow(a, dim, start, length);
}

TensorDataPtr CPUBackend::randn(const std::vector<size_t>& shape, DType dtype) {
    return cpu::randn(shape, dtype);
}

TensorDataPtr CPUBackend::zeros(const std::vector<size_t>& shape, DType dtype) {
    return cpu::zeros(shape, dtype);
}

TensorDataPtr CPUBackend::ones(const std::vector<size_t>& shape, DType dtype) {
    return cpu::ones(shape, dtype);
}

TensorDataPtr CPUBackend::softmax_cross_entropy(const TensorDataPtr& logits, const TensorDataPtr& targets) {
    return cpu::softmax_cross_entropy(logits, targets);
}

TensorDataPtr CPUBackend::softmax_cross_entropy_backward(const TensorDataPtr& grad_out, const TensorDataPtr& logits, const TensorDataPtr& targets) {
    return cpu::softmax_cross_entropy_backward(grad_out, logits, targets);
}

} // namespace phoenix
