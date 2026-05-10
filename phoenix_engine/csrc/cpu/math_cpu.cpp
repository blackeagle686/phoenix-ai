#include "cpu_kernels.h"
#include <stdexcept>
#include <random>
#include <cmath>

namespace phoenix {
namespace cpu {

template <typename T>
void add_impl(const T* a, const T* b, T* out, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        out[i] = a[i] + b[i];
    }
}

TensorDataPtr add(const TensorDataPtr& a, const TensorDataPtr& b) {
    bool can_broadcast = false;
    if (a->shape() != b->shape()) {
        // Basic broadcasting: if b is [1, N] and a is [M, N]
        if (a->shape().size() == 2 && b->shape().size() == 2 &&
            b->shape()[0] == 1 && a->shape()[1] == b->shape()[1]) {
            can_broadcast = true;
        } else {
            throw std::invalid_argument("Shapes must match for addition (Broadcasting not fully implemented yet)");
        }
    }

    auto out = std::make_shared<TensorData>(a->shape(), a->dtype(), Device::CPU);

    if (a->dtype() == DType::Float32) {
        float* a_ptr = static_cast<float*>(a->data());
        float* b_ptr = static_cast<float*>(b->data());
        float* out_ptr = static_cast<float*>(out->data());
        
        if (can_broadcast) {
            size_t M = a->shape()[0];
            size_t N = a->shape()[1];
            for (size_t i = 0; i < M; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    float a_val = a_ptr[i * a->strides()[0] + j * a->strides()[1]];
                    float b_val = b_ptr[0 * b->strides()[0] + j * b->strides()[1]]; // b is [1, N]
                    out_ptr[i * out->strides()[0] + j * out->strides()[1]] = a_val + b_val;
                }
            }
        } else {
            if (!a->is_contiguous() || !b->is_contiguous()) {
                throw std::runtime_error("Addition of non-contiguous tensors not fully implemented. Call .contiguous() first.");
            }
            add_impl(a_ptr, b_ptr, out_ptr, a->size());
        }
    } else {
        throw std::runtime_error("Addition only implemented for Float32 right now");
    }

    return out;
}

template <typename T>
void mul_impl(const T* a, const T* b, T* out, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        out[i] = a[i] * b[i];
    }
}

TensorDataPtr multiply(const TensorDataPtr& a, const TensorDataPtr& b) {
    if (a->shape() != b->shape()) {
        throw std::invalid_argument("Shapes must match for multiplication");
    }
    if (a->dtype() != b->dtype()) {
        throw std::invalid_argument("DTypes must match for multiplication");
    }

    auto out = std::make_shared<TensorData>(a->shape(), a->dtype(), Device::CPU);

    if (a->dtype() == DType::Float32) {
        if (!a->is_contiguous() || !b->is_contiguous()) {
            throw std::runtime_error("Multiplication of non-contiguous tensors not fully implemented. Call .contiguous() first.");
        }
        mul_impl(static_cast<const float*>(a->data()), 
                 static_cast<const float*>(b->data()), 
                 static_cast<float*>(out->data()), a->size());
    } else {
        throw std::runtime_error("Multiplication only implemented for Float32 right now");
    }

    return out;
}

template <typename T>
void multiply_scalar_impl(const T* a, T scalar, T* out, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        out[i] = a[i] * scalar;
    }
}

TensorDataPtr multiply_scalar(const TensorDataPtr& a, float scalar) {
    if (!a->is_contiguous()) {
        throw std::runtime_error("multiply_scalar requires contiguous memory for now.");
    }

    auto out = std::make_shared<TensorData>(a->shape(), a->dtype(), Device::CPU);

    if (a->dtype() == DType::Float32) {
        multiply_scalar_impl(static_cast<const float*>(a->data()), scalar, 
                             static_cast<float*>(out->data()), a->size());
    } else {
        throw std::runtime_error("multiply_scalar only implemented for Float32 right now");
    }

    return out;
}

template <typename T>
void gemm_impl(const T* a, const T* b, T* out, size_t m, size_t k, size_t n,
               size_t stride_a0, size_t stride_a1,
               size_t stride_b0, size_t stride_b1,
               size_t stride_out0, size_t stride_out1) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            T sum = 0;
            for (size_t p = 0; p < k; ++p) {
                sum += a[i * stride_a0 + p * stride_a1] * b[p * stride_b0 + j * stride_b1];
            }
            out[i * stride_out0 + j * stride_out1] = sum;
        }
    }
}

TensorDataPtr gemm(const TensorDataPtr& a, const TensorDataPtr& b) {
    if (a->shape().size() < 2 || b->shape().size() < 2) {
        throw std::invalid_argument("GEMM requires at least 2D tensors");
    }
    if (a->shape().size() != b->shape().size()) {
        throw std::invalid_argument("GEMM requires tensors to have the same number of dimensions for now");
    }
    
    int rank = a->shape().size();
    size_t m = a->shape()[rank - 2];
    size_t k = a->shape()[rank - 1];
    
    if (b->shape()[rank - 2] != k) {
        throw std::invalid_argument("Inner dimensions must match for GEMM");
    }
    size_t n = b->shape()[rank - 1];

    size_t batch_size = 1;
    for (int i = 0; i < rank - 2; ++i) {
        if (a->shape()[i] != b->shape()[i]) {
            throw std::invalid_argument("Batch dimensions must match for Batched GEMM");
        }
        batch_size *= a->shape()[i];
    }

    std::vector<size_t> out_shape = a->shape();
    out_shape[rank - 2] = m;
    out_shape[rank - 1] = n;

    auto out = std::make_shared<TensorData>(out_shape, a->dtype(), Device::CPU);

    if (a->dtype() == DType::Float32) {
        const float* a_ptr = static_cast<const float*>(a->data());
        const float* b_ptr = static_cast<const float*>(b->data());
        float* out_ptr = static_cast<float*>(out->data());

        std::vector<size_t> batch_coords(rank > 2 ? rank - 2 : 1, 0);

        for (size_t b_idx = 0; b_idx < batch_size; ++b_idx) {
            size_t offset_a = 0, offset_b = 0, offset_out = 0;
            if (rank > 2) {
                for (int d = 0; d < rank - 2; ++d) {
                    offset_a += batch_coords[d] * a->strides()[d];
                    offset_b += batch_coords[d] * b->strides()[d];
                    offset_out += batch_coords[d] * out->strides()[d];
                }
            }

            gemm_impl(a_ptr + offset_a, 
                      b_ptr + offset_b, 
                      out_ptr + offset_out, m, k, n,
                      a->strides()[rank - 2], a->strides()[rank - 1],
                      b->strides()[rank - 2], b->strides()[rank - 1],
                      out->strides()[rank - 2], out->strides()[rank - 1]);

            // Update odometer
            if (rank > 2) {
                for (int d = rank - 3; d >= 0; --d) {
                    batch_coords[d]++;
                    if (batch_coords[d] < a->shape()[d]) break;
                    batch_coords[d] = 0;
                }
            }
        }
    } else {
        throw std::runtime_error("GEMM only implemented for Float32 right now");
    }

    return out;
}

template <typename T>
void sub_impl(const T* a, const T* b, T* out, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        out[i] = a[i] - b[i];
    }
}

TensorDataPtr sub(const TensorDataPtr& a, const TensorDataPtr& b) {
    if (a->shape() != b->shape()) throw std::invalid_argument("Shapes must match for subtraction");
    if (a->dtype() != b->dtype()) throw std::invalid_argument("DTypes must match for subtraction");

    auto out = std::make_shared<TensorData>(a->shape(), a->dtype(), Device::CPU);
    if (a->dtype() == DType::Float32) {
        if (!a->is_contiguous() || !b->is_contiguous()) {
            throw std::runtime_error("Subtraction of non-contiguous tensors not fully implemented. Call .contiguous() first.");
        }
        sub_impl(static_cast<const float*>(a->data()), static_cast<const float*>(b->data()), static_cast<float*>(out->data()), a->size());
    } else {
        throw std::runtime_error("Subtraction only implemented for Float32");
    }
    return out;
}

template <typename T>
void sum_impl(const T* a, T* out, size_t size) {
    T sum_val = 0;
    for (size_t i = 0; i < size; ++i) {
        sum_val += a[i];
    }
    out[0] = sum_val;
}

TensorDataPtr sum(const TensorDataPtr& a) {
    auto out = std::make_shared<TensorData>(std::vector<size_t>{1}, a->dtype(), Device::CPU);
    if (a->dtype() == DType::Float32) {
        sum_impl(static_cast<const float*>(a->data()), static_cast<float*>(out->data()), a->size());
    } else {
        throw std::runtime_error("Sum only implemented for Float32");
    }
    return out;
}

template <typename T>
void relu_impl(const T* a, T* out, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        out[i] = a[i] > 0 ? a[i] : 0;
    }
}

TensorDataPtr relu(const TensorDataPtr& a) {
    auto out = std::make_shared<TensorData>(a->shape(), a->dtype(), Device::CPU);
    if (a->dtype() == DType::Float32) {
        relu_impl(static_cast<const float*>(a->data()), static_cast<float*>(out->data()), a->size());
    } else {
        throw std::runtime_error("ReLU only implemented for Float32");
    }
    return out;
}

template <typename T>
void softmax_impl(const T* a, T* out, size_t outer_size, size_t inner_size) {
    for (size_t i = 0; i < outer_size; ++i) {
        const T* a_row = a + i * inner_size;
        T* out_row = out + i * inner_size;
        
        // Find max for numerical stability
        T max_val = a_row[0];
        for (size_t j = 1; j < inner_size; ++j) {
            if (a_row[j] > max_val) max_val = a_row[j];
        }
        
        // Compute exp and sum
        T sum = 0;
        for (size_t j = 0; j < inner_size; ++j) {
            out_row[j] = std::exp(a_row[j] - max_val);
            sum += out_row[j];
        }
        
        // Normalize
        for (size_t j = 0; j < inner_size; ++j) {
            out_row[j] /= sum;
        }
    }
}

TensorDataPtr softmax(const TensorDataPtr& a) {
    if (!a->is_contiguous()) {
        throw std::runtime_error("Softmax requires a contiguous tensor. Call .contiguous() first.");
    }
    
    auto out = std::make_shared<TensorData>(a->shape(), a->dtype(), Device::CPU);
    
    if (a->dtype() == DType::Float32) {
        size_t inner_size = a->shape().back();
        size_t outer_size = a->size() / inner_size;
        
        softmax_impl(static_cast<const float*>(a->data()), static_cast<float*>(out->data()), outer_size, inner_size);
    } else {
        throw std::runtime_error("Softmax only implemented for Float32");
    }
    return out;
}

template <typename T>
void layernorm_impl(const T* a, const T* weight, const T* bias, T* out, size_t outer_size, size_t inner_size, float eps) {
    for (size_t i = 0; i < outer_size; ++i) {
        const T* a_row = a + i * inner_size;
        T* out_row = out + i * inner_size;
        
        // Mean
        T sum = 0;
        for (size_t j = 0; j < inner_size; ++j) {
            sum += a_row[j];
        }
        T mean = sum / inner_size;
        
        // Variance
        T var = 0;
        for (size_t j = 0; j < inner_size; ++j) {
            T diff = a_row[j] - mean;
            var += diff * diff;
        }
        var /= inner_size;
        
        // Normalize and apply weight/bias
        T stddev_inv = 1.0f / std::sqrt(var + eps);
        for (size_t j = 0; j < inner_size; ++j) {
            T normalized = (a_row[j] - mean) * stddev_inv;
            out_row[j] = normalized * (weight ? weight[j] : 1.0f) + (bias ? bias[j] : 0.0f);
        }
    }
}

TensorDataPtr layernorm(const TensorDataPtr& a, const TensorDataPtr& weight, const TensorDataPtr& bias, float eps) {
    if (!a->is_contiguous()) {
        throw std::runtime_error("LayerNorm requires a contiguous tensor.");
    }
    
    auto out = std::make_shared<TensorData>(a->shape(), a->dtype(), Device::CPU);
    
    if (a->dtype() == DType::Float32) {
        size_t inner_size = a->shape().back();
        size_t outer_size = a->size() / inner_size;
        
        const float* w_ptr = weight ? static_cast<const float*>(weight->data()) : nullptr;
        const float* b_ptr = bias ? static_cast<const float*>(bias->data()) : nullptr;
        
        layernorm_impl(static_cast<const float*>(a->data()), w_ptr, b_ptr, static_cast<float*>(out->data()), outer_size, inner_size, eps);
    } else {
        throw std::runtime_error("LayerNorm only implemented for Float32");
    }
    return out;
}

TensorDataPtr embedding(const TensorDataPtr& weight, const TensorDataPtr& indices) {
    if (!weight->is_contiguous() || !indices->is_contiguous()) {
        throw std::runtime_error("Embedding requires contiguous tensors.");
    }
    
    size_t embed_dim = weight->shape().back();
    size_t num_embeddings = weight->size() / embed_dim;
    
    std::vector<size_t> out_shape = indices->shape();
    out_shape.push_back(embed_dim);
    
    auto out = std::make_shared<TensorData>(out_shape, weight->dtype(), Device::CPU);
    
    if (weight->dtype() == DType::Float32 && indices->dtype() == DType::Float32) {
        const float* w_ptr = static_cast<const float*>(weight->data());
        const float* idx_ptr = static_cast<const float*>(indices->data());
        float* out_ptr = static_cast<float*>(out->data());
        
        for (size_t i = 0; i < indices->size(); ++i) {
            size_t idx = static_cast<size_t>(idx_ptr[i]);
            if (idx >= num_embeddings) throw std::out_of_range("Embedding index out of bounds");
            
            for (size_t j = 0; j < embed_dim; ++j) {
                out_ptr[i * embed_dim + j] = w_ptr[idx * embed_dim + j];
            }
        }
    } else {
        throw std::runtime_error("Embedding only implemented for Float32 right now");
    }
    return out;
}

TensorDataPtr randn(const std::vector<size_t>& shape, DType dtype) {
    auto out = std::make_shared<TensorData>(shape, dtype, Device::CPU);
    
    if (dtype == DType::Float32) {
        float* out_ptr = static_cast<float*>(out->data());
        size_t size = out->size();
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> d(0.0f, 1.0f);
        
        for (size_t i = 0; i < size; ++i) {
            out_ptr[i] = d(gen);
        }
    } else {
        throw std::runtime_error("randn only implemented for Float32");
    }
    
    return out;
}

template <typename T>
void fill_impl(T* data, size_t size, T value) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = value;
    }
}

TensorDataPtr zeros(const std::vector<size_t>& shape, DType dtype) {
    auto out = std::make_shared<TensorData>(shape, dtype, Device::CPU);
    if (dtype == DType::Float32) {
        fill_impl(static_cast<float*>(out->data()), out->size(), 0.0f);
    } else if (dtype == DType::Int32) {
        fill_impl(static_cast<int32_t*>(out->data()), out->size(), 0);
    } else {
        throw std::runtime_error("zeros only implemented for Float32 and Int32");
    }
    return out;
}

TensorDataPtr ones(const std::vector<size_t>& shape, DType dtype) {
    auto out = std::make_shared<TensorData>(shape, dtype, Device::CPU);
    if (dtype == DType::Float32) {
        fill_impl(static_cast<float*>(out->data()), out->size(), 1.0f);
    } else if (dtype == DType::Int32) {
        fill_impl(static_cast<int32_t*>(out->data()), out->size(), 1);
    } else {
        throw std::runtime_error("ones only implemented for Float32 and Int32");
    }
    return out;
}

} // namespace cpu
} // namespace phoenix
