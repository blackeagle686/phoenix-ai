#pragma once

#include <vector>
#include <memory>
#include <stdexcept>
#include <iostream>
#include "types.h"

namespace phoenix {

class TensorData : public std::enable_shared_from_this<TensorData> {
public:
    TensorData(const std::vector<size_t>& shape, DType dtype, Device device);
    // Special constructor for creating views (sharing memory)
    TensorData(const std::vector<size_t>& shape, const std::vector<size_t>& strides, 
               DType dtype, Device device, std::shared_ptr<void> storage, size_t offset, bool is_contiguous = false);
    ~TensorData();

    // Prevent copying (we pass pointers around)
    TensorData(const TensorData&) = delete;
    TensorData& operator=(const TensorData&) = delete;

    // Getters
    void* data() const { return data_; }
    const std::vector<size_t>& shape() const { return shape_; }
    const std::vector<size_t>& strides() const { return strides_; }
    DType dtype() const { return dtype_; }
    Device device() const { return device_; }
    size_t size() const { return size_; }
    size_t num_bytes() const { return size_ * element_size(dtype_); }
    bool is_contiguous() const { return is_contiguous_; }
    size_t offset() const { return offset_; }

    // O(1) View Operations
    std::shared_ptr<TensorData> view(const std::vector<size_t>& new_shape);
    std::shared_ptr<TensorData> transpose(size_t dim0, size_t dim1);
    std::shared_ptr<TensorData> permute(const std::vector<size_t>& dims);
    std::shared_ptr<TensorData> contiguous();

    // Data Access
    std::vector<float> to_float_list();
    std::vector<int32_t> to_int_list();
    std::string to_string() const;

private:
    void calculate_strides();

private:
    std::shared_ptr<void> storage_;
    size_t offset_;
    void* data_; // Cached pointer: storage_.get() + offset_
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    DType dtype_;
    Device device_;
    size_t size_;
    bool is_contiguous_;
    
    void allocate();
    void deallocate();
};

// Using shared_ptr to handle reference counting across Python and C++
using TensorDataPtr = std::shared_ptr<TensorData>;

} // namespace phoenix
