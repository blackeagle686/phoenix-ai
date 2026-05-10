#include "tensor_data.h"
#include <numeric>
#include <cstdlib>
#include <sstream>

namespace phoenix {

TensorData::TensorData(const std::vector<size_t>& shape, DType dtype, Device device)
    : shape_(shape), dtype_(dtype), device_(device), data_(nullptr), is_contiguous_(true) {
    
    size_ = 1;
    for (auto dim : shape) {
        size_ *= dim;
    }

    calculate_strides();
    allocate();
}

void TensorData::calculate_strides() {
    strides_.resize(shape_.size());
    size_t stride = 1;
    for (int i = shape_.size() - 1; i >= 0; --i) {
        strides_[i] = stride;
        stride *= shape_[i];
    }
}

TensorData::~TensorData() {
    deallocate();
}

void TensorData::allocate() {
    if (size_ == 0) return;

    size_t bytes = num_bytes();
    
    if (device_ == Device::CPU) {
        // Standard aligned malloc can be added later, using malloc for now
        data_ = std::malloc(bytes);
        if (!data_) {
            throw std::runtime_error("Failed to allocate memory on CPU.");
        }
    } else if (device_ == Device::CUDA) {
        throw std::runtime_error("CUDA allocation not yet implemented.");
    } else {
        throw std::invalid_argument("Unknown device type.");
    }
}

void TensorData::deallocate() {
    if (!data_) return;

    if (device_ == Device::CPU) {
        std::free(data_);
    } else if (device_ == Device::CUDA) {
        // CUDA free logic later
    }
    data_ = nullptr;
}

std::string TensorData::to_string() const {
    std::ostringstream oss;
    oss << "TensorData(shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        oss << shape_[i] << (i < shape_.size() - 1 ? ", " : "");
    }
    oss << "], device=" << device_to_string(device_) << ", ptr=" << data_ << ")";
    return oss.str();
}

} // namespace phoenix
