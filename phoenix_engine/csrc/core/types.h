#pragma once

#include <cstdint>
#include <string>

namespace phoenix {

enum class Device {
    CPU,
    CUDA
};

enum class DType {
    Float32,
    Float64,
    Int32,
    Int64
};

inline size_t element_size(DType dtype) {
    switch (dtype) {
        case DType::Float32: return 4;
        case DType::Float64: return 8;
        case DType::Int32:   return 4;
        case DType::Int64:   return 8;
        default: return 0;
    }
}

inline std::string device_to_string(Device d) {
    switch (d) {
        case Device::CPU: return "CPU";
        case Device::CUDA: return "CUDA";
        default: return "Unknown";
    }
}

} // namespace phoenix
