#pragma once

#include <vector>
#include <memory>
#include <stdexcept>
#include <iostream>
#include "types.h"

namespace phoenix {

class TensorData {
public:
    TensorData(const std::vector<size_t>& shape, DType dtype, Device device);
    ~TensorData();

    // Prevent copying (we pass pointers around)
    TensorData(const TensorData&) = delete;
    TensorData& operator=(const TensorData&) = delete;

    // Getters
    void* data() const { return data_; }
    const std::vector<size_t>& shape() const { return shape_; }
    DType dtype() const { return dtype_; }
    Device device() const { return device_; }
    size_t size() const { return size_; }
    size_t num_bytes() const { return size_ * element_size(dtype_); }

    // Helpers
    std::string to_string() const;

private:
    void* data_;
    std::vector<size_t> shape_;
    DType dtype_;
    Device device_;
    size_t size_;
    
    void allocate();
    void deallocate();
};

// Using shared_ptr to handle reference counting across Python and C++
using TensorDataPtr = std::shared_ptr<TensorData>;

} // namespace phoenix
