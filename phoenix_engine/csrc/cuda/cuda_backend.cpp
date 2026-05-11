#include "cuda_backend.h"
#include <stdexcept>
#include <string>

namespace phoenix {

#ifdef USE_CUDA

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err))); \
        } \
    } while (0)

void* CUDABackend::allocate(size_t num_bytes) {
    void* ptr;
    CHECK_CUDA(cudaMalloc(&ptr, num_bytes));
    return ptr;
}

void CUDABackend::deallocate(void* ptr) {
    if (ptr) {
        cudaFree(ptr);
    }
}

#else

void* CUDABackend::allocate(size_t num_bytes) {
    throw std::runtime_error("CUDABackend::allocate called but USE_CUDA is not defined.");
}

void CUDABackend::deallocate(void* ptr) {
    // Nothing to do
}

#endif

// Stubs for now
TensorDataPtr CUDABackend::add(const TensorDataPtr& a, const TensorDataPtr& b) { throw std::runtime_error("CUDA add not implemented"); }
TensorDataPtr CUDABackend::add_scalar(const TensorDataPtr& a, float scalar) { throw std::runtime_error("CUDA add_scalar not implemented"); }
TensorDataPtr CUDABackend::sub(const TensorDataPtr& a, const TensorDataPtr& b) { throw std::runtime_error("CUDA sub not implemented"); }
TensorDataPtr CUDABackend::multiply(const TensorDataPtr& a, const TensorDataPtr& b) { throw std::runtime_error("CUDA multiply not implemented"); }
TensorDataPtr CUDABackend::multiply_scalar(const TensorDataPtr& a, float scalar) { throw std::runtime_error("CUDA multiply_scalar not implemented"); }
TensorDataPtr CUDABackend::divide(const TensorDataPtr& a, const TensorDataPtr& b) { throw std::runtime_error("CUDA divide not implemented"); }
TensorDataPtr CUDABackend::divide_scalar(const TensorDataPtr& a, float scalar) { throw std::runtime_error("CUDA divide_scalar not implemented"); }
TensorDataPtr CUDABackend::sqrt(const TensorDataPtr& a) { throw std::runtime_error("CUDA sqrt not implemented"); }
TensorDataPtr CUDABackend::sum(const TensorDataPtr& a) { throw std::runtime_error("CUDA sum not implemented"); }

TensorDataPtr CUDABackend::relu(const TensorDataPtr& a) { throw std::runtime_error("CUDA relu not implemented"); }
TensorDataPtr CUDABackend::softmax(const TensorDataPtr& a) { throw std::runtime_error("CUDA softmax not implemented"); }
TensorDataPtr CUDABackend::layernorm(const TensorDataPtr& a, const TensorDataPtr& weight, const TensorDataPtr& bias, float eps) { throw std::runtime_error("CUDA layernorm not implemented"); }

TensorDataPtr CUDABackend::gemm(const TensorDataPtr& a, const TensorDataPtr& b) { throw std::runtime_error("CUDA gemm not implemented"); }

TensorDataPtr CUDABackend::embedding_forward(const TensorDataPtr& indices, const TensorDataPtr& weight) { throw std::runtime_error("CUDA embedding_forward not implemented"); }
TensorDataPtr CUDABackend::masked_fill(const TensorDataPtr& a, const TensorDataPtr& mask, float value) { throw std::runtime_error("CUDA masked_fill not implemented"); }
TensorDataPtr CUDABackend::tril(const std::vector<size_t>& shape) { throw std::runtime_error("CUDA tril not implemented"); }
TensorDataPtr CUDABackend::narrow(const TensorDataPtr& a, size_t dim, size_t start, size_t length) { throw std::runtime_error("CUDA narrow not implemented"); }

TensorDataPtr CUDABackend::randn(const std::vector<size_t>& shape, DType dtype) { throw std::runtime_error("CUDA randn not implemented"); }
TensorDataPtr CUDABackend::zeros(const std::vector<size_t>& shape, DType dtype) { throw std::runtime_error("CUDA zeros not implemented"); }
TensorDataPtr CUDABackend::ones(const std::vector<size_t>& shape, DType dtype) { throw std::runtime_error("CUDA ones not implemented"); }

TensorDataPtr CUDABackend::softmax_cross_entropy(const TensorDataPtr& logits, const TensorDataPtr& targets) { throw std::runtime_error("CUDA softmax_cross_entropy not implemented"); }
TensorDataPtr CUDABackend::softmax_cross_entropy_backward(const TensorDataPtr& grad_out, const TensorDataPtr& logits, const TensorDataPtr& targets) { throw std::runtime_error("CUDA softmax_cross_entropy_backward not implemented"); }

} // namespace phoenix
