#include "quantum_backend.h"
#include <cstdlib>
#include <stdexcept>
#include <iostream>

namespace phoenix {

void* QuantumBackend::allocate(size_t num_bytes) {
    std::cout << "[QuantumBackend] Allocating " << num_bytes << " bytes for state-vector simulation." << std::endl;
    void* ptr = std::malloc(num_bytes);
    if (!ptr) throw std::runtime_error("QuantumBackend: Failed to allocate " + std::to_string(num_bytes) + " bytes.");
    return ptr;
}

void QuantumBackend::deallocate(void* ptr) {
    std::free(ptr);
}

// Simulated Ops (State-vector math)
TensorDataPtr QuantumBackend::add(const TensorDataPtr& a, const TensorDataPtr& b) { throw std::runtime_error("Quantum add (Superposition) not implemented yet"); }
TensorDataPtr QuantumBackend::add_scalar(const TensorDataPtr& a, float scalar) { throw std::runtime_error("Quantum add_scalar not implemented yet"); }
TensorDataPtr QuantumBackend::sub(const TensorDataPtr& a, const TensorDataPtr& b) { throw std::runtime_error("Quantum sub not implemented yet"); }
TensorDataPtr QuantumBackend::multiply(const TensorDataPtr& a, const TensorDataPtr& b) { throw std::runtime_error("Quantum multiply (Entanglement?) not implemented yet"); }
TensorDataPtr QuantumBackend::multiply_scalar(const TensorDataPtr& a, float scalar) { throw std::runtime_error("Quantum multiply_scalar not implemented yet"); }
TensorDataPtr QuantumBackend::divide(const TensorDataPtr& a, const TensorDataPtr& b) { throw std::runtime_error("Quantum divide not implemented yet"); }
TensorDataPtr QuantumBackend::divide_scalar(const TensorDataPtr& a, float scalar) { throw std::runtime_error("Quantum divide_scalar not implemented yet"); }
TensorDataPtr QuantumBackend::sqrt(const TensorDataPtr& a) { throw std::runtime_error("Quantum sqrt not implemented yet"); }
TensorDataPtr QuantumBackend::sum(const TensorDataPtr& a) { throw std::runtime_error("Quantum sum (Collapse?) not implemented yet"); }

TensorDataPtr QuantumBackend::relu(const TensorDataPtr& a) { throw std::runtime_error("Non-linear ReLU is not natively quantum-compatible."); }
TensorDataPtr QuantumBackend::softmax(const TensorDataPtr& a) { throw std::runtime_error("Softmax not implemented for QuantumBackend."); }
TensorDataPtr QuantumBackend::layernorm(const TensorDataPtr& a, const TensorDataPtr& weight, const TensorDataPtr& bias, float eps) { throw std::runtime_error("LayerNorm not implemented for QuantumBackend."); }

TensorDataPtr QuantumBackend::gemm(const TensorDataPtr& a, const TensorDataPtr& b) {
    // In Quantum, GEMM is often a Unitary Gate application
    throw std::runtime_error("Quantum GEMM (Unitary Gate Application) not implemented yet");
}

TensorDataPtr QuantumBackend::embedding_forward(const TensorDataPtr& indices, const TensorDataPtr& weight) { throw std::runtime_error("Quantum embedding not implemented"); }
TensorDataPtr QuantumBackend::masked_fill(const TensorDataPtr& a, const TensorDataPtr& mask, float value) { throw std::runtime_error("Quantum masked_fill not implemented"); }
TensorDataPtr QuantumBackend::tril(const std::vector<size_t>& shape) { throw std::runtime_error("Quantum tril not implemented"); }
TensorDataPtr QuantumBackend::narrow(const TensorDataPtr& a, size_t dim, size_t start, size_t length) { throw std::runtime_error("Quantum narrow not implemented"); }

TensorDataPtr QuantumBackend::randn(const std::vector<size_t>& shape, DType dtype) { throw std::runtime_error("Quantum randn not implemented"); }
TensorDataPtr QuantumBackend::zeros(const std::vector<size_t>& shape, DType dtype) { throw std::runtime_error("Quantum zeros (|0...0> state) not implemented yet"); }
TensorDataPtr QuantumBackend::ones(const std::vector<size_t>& shape, DType dtype) { throw std::runtime_error("Quantum ones not implemented"); }

TensorDataPtr QuantumBackend::softmax_cross_entropy(const TensorDataPtr& logits, const TensorDataPtr& targets) { throw std::runtime_error("Quantum Loss not implemented"); }
TensorDataPtr QuantumBackend::softmax_cross_entropy_backward(const TensorDataPtr& grad_out, const TensorDataPtr& logits, const TensorDataPtr& targets) { throw std::runtime_error("Quantum Loss Backward not implemented"); }

} // namespace phoenix
