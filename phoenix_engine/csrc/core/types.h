#pragma once

#include <cstdint>
#include <string>

namespace phoenix {

enum class Device {
    CPU,
    CUDA,
    QUANTUM
};

enum class DType {
    Float32,
    Float64,
    Int32,
    Int64,
    Complex64,
    Complex128
};

inline size_t element_size(DType dtype) {
    switch (dtype) {
        case DType::Float32: return 4;
        case DType::Float64: return 8;
        case DType::Int32:   return 4;
        case DType::Int64:   return 8;
        case DType::Complex64: return 8;
        case DType::Complex128: return 16;
        default: return 0;
    }
}

inline std::string device_to_string(Device d) {
    switch (d) {
        case Device::CPU: return "CPU";
        case Device::CUDA: return "CUDA";
        case Device::QUANTUM: return "QUANTUM";
        default: return "Unknown";
    }
}

} // namespace phoenix
