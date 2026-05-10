#include "cpu_kernels.h"
#include <stdexcept>
#include <random>

namespace phoenix {
namespace cpu {

template <typename T>
void add_impl(const T* a, const T* b, T* out, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        out[i] = a[i] + b[i];
    }
}

TensorDataPtr add(const TensorDataPtr& a, const TensorDataPtr& b) {
    if (a->shape() != b->shape()) {
        throw std::invalid_argument("Shapes must match for addition");
    }
    if (a->dtype() != b->dtype()) {
        throw std::invalid_argument("DTypes must match for addition");
    }

    auto out = std::make_shared<TensorData>(a->shape(), a->dtype(), Device::CPU);

    if (a->dtype() == DType::Float32) {
        add_impl(static_cast<const float*>(a->data()), 
                 static_cast<const float*>(b->data()), 
                 static_cast<float*>(out->data()), a->size());
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
        mul_impl(static_cast<const float*>(a->data()), 
                 static_cast<const float*>(b->data()), 
                 static_cast<float*>(out->data()), a->size());
    } else {
        throw std::runtime_error("Multiplication only implemented for Float32 right now");
    }

    return out;
}

template <typename T>
void gemm_impl(const T* a, const T* b, T* out, size_t m, size_t k, size_t n) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            T sum = 0;
            for (size_t p = 0; p < k; ++p) {
                sum += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = sum;
        }
    }
}

TensorDataPtr gemm(const TensorDataPtr& a, const TensorDataPtr& b) {
    if (a->shape().size() != 2 || b->shape().size() != 2) {
        throw std::invalid_argument("GEMM requires 2D tensors");
    }
    if (a->shape()[1] != b->shape()[0]) {
        throw std::invalid_argument("Inner dimensions must match for GEMM");
    }
    if (a->dtype() != b->dtype()) {
        throw std::invalid_argument("DTypes must match for GEMM");
    }

    size_t m = a->shape()[0];
    size_t k = a->shape()[1];
    size_t n = b->shape()[1];

    auto out = std::make_shared<TensorData>(std::vector<size_t>{m, n}, a->dtype(), Device::CPU);

    if (a->dtype() == DType::Float32) {
        gemm_impl(static_cast<const float*>(a->data()), 
                  static_cast<const float*>(b->data()), 
                  static_cast<float*>(out->data()), m, k, n);
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

TensorDataPtr randn(const std::vector<size_t>& shape, DType dtype) {
    auto out = std::make_shared<TensorData>(shape, dtype, Device::CPU);
    
    if (dtype == DType::Float32) {
        float* data = static_cast<float*>(out->data());
        // Using a static generator is fast enough for now, but not thread-safe.
        static std::random_device rd;
        static std::mt19937 gen{rd()};
        std::normal_distribution<float> d(0.0f, 1.0f);
        
        for (size_t i = 0; i < out->size(); ++i) {
            data[i] = d(gen);
        }
    } else {
         throw std::runtime_error("randn only implemented for Float32");
    }
    
    return out;
}

} // namespace cpu
} // namespace phoenix
