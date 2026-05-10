#include "tensor_data.h"
#include <numeric>
#include <cstdlib>
#include <sstream>

namespace phoenix {

TensorData::TensorData(const std::vector<size_t>& shape, DType dtype, Device device)
    : shape_(shape), dtype_(dtype), device_(device), offset_(0), is_contiguous_(true) {
    
    size_ = 1;
    for (auto dim : shape) size_ *= dim;

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

TensorData::TensorData(const std::vector<size_t>& shape, const std::vector<size_t>& strides, 
                       DType dtype, Device device, std::shared_ptr<void> storage, size_t offset)
    : shape_(shape), strides_(strides), dtype_(dtype), device_(device), 
      storage_(storage), offset_(offset), is_contiguous_(false) {
    
    size_ = 1;
    for (auto dim : shape) size_ *= dim;
    data_ = (char*)storage_.get() + offset;
}

TensorData::~TensorData() {
    // std::shared_ptr handles deallocation
}

void TensorData::allocate() {
    if (size_ == 0) return;
    size_t bytes = num_bytes();
    
    if (device_ == Device::CPU) {
        void* ptr = std::malloc(bytes);
        if (!ptr) throw std::runtime_error("Failed to allocate memory on CPU.");
        storage_ = std::shared_ptr<void>(ptr, std::free);
        data_ = ptr;
    } else {
        throw std::runtime_error("CUDA allocation not yet implemented.");
    }
}

std::shared_ptr<TensorData> TensorData::view(const std::vector<size_t>& new_shape) {
    size_t new_size = 1;
    for (auto d : new_shape) new_size *= d;
    if (new_size != size_) throw std::runtime_error("Incompatible size for view");
    if (!is_contiguous_) throw std::runtime_error("View only supported on contiguous tensors currently");

    // For a contiguous view, we can just calculate new row-major strides
    std::vector<size_t> new_strides(new_shape.size());
    size_t s = 1;
    for (int i = new_shape.size() - 1; i >= 0; --i) {
        new_strides[i] = s;
        s *= new_shape[i];
    }

    return std::make_shared<TensorData>(new_shape, new_strides, dtype_, device_, storage_, offset_);
}

std::shared_ptr<TensorData> TensorData::transpose(size_t dim0, size_t dim1) {
    if (dim0 >= shape_.size() || dim1 >= shape_.size()) throw std::out_of_range("Dimension out of range");

    std::vector<size_t> new_shape = shape_;
    std::vector<size_t> new_strides = strides_;
    std::swap(new_shape[dim0], new_shape[dim1]);
    std::swap(new_strides[dim0], new_strides[dim1]);

    return std::make_shared<TensorData>(new_shape, new_strides, dtype_, device_, storage_, offset_);
}

std::shared_ptr<TensorData> TensorData::permute(const std::vector<size_t>& dims) {
    if (dims.size() != shape_.size()) throw std::runtime_error("Permute dims must match tensor rank");

    std::vector<size_t> new_shape(dims.size());
    std::vector<size_t> new_strides(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        new_shape[i] = shape_[dims[i]];
        new_strides[i] = strides_[dims[i]];
    }

    return std::make_shared<TensorData>(new_shape, new_strides, dtype_, device_, storage_, offset_);
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
