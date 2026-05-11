#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include "core/tensor_data.h"
#include "core/dispatcher.h"
#include "cpu/cpu_backend.h"
#include "cuda/cuda_backend.h"

namespace py = pybind11;
using namespace phoenix;

// Helper to get backend from a tensor
Backend* get_backend(const TensorDataPtr& a) {
    return Dispatcher::instance().get_backend(a->device());
}

std::string hello() {
    return "Hello from Phoenix C++ Engine!";
}

PYBIND11_MODULE(_phoenix_backend, m) {
    // Register backends
    Dispatcher::instance().register_backend(Device::CPU, std::make_unique<CPUBackend>());
    Dispatcher::instance().register_backend(Device::CUDA, std::make_unique<CUDABackend>());

    m.doc() = "Phoenix Engine Low-Level C++ Backend with Dispatcher Architecture";
    m.def("hello", &hello, "A function that returns a hello string");

    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("CUDA", Device::CUDA)
        .export_values();

    py::enum_<DType>(m, "DType")
        .value("Float32", DType::Float32)
        .value("Float64", DType::Float64)
        .value("Int32", DType::Int32)
        .value("Int64", DType::Int64)
        .export_values();

    py::class_<TensorData, std::shared_ptr<TensorData>>(m, "TensorData")
        .def(py::init<const std::vector<size_t>&, DType, Device>(), 
             py::arg("shape"), py::arg("dtype"), py::arg("device") = Device::CPU)
        .def("shape", &TensorData::shape)
        .def("strides", &TensorData::strides)
        .def("dtype", &TensorData::dtype)
        .def("device", &TensorData::device)
        .def("size", &TensorData::size)
        .def("num_bytes", &TensorData::num_bytes)
        .def("view", &TensorData::view)
        .def("transpose", &TensorData::transpose)
        .def("permute", &TensorData::permute)
        .def("contiguous", &TensorData::contiguous)
        .def("to_float_list", &TensorData::to_float_list)
        .def("to_int_list", &TensorData::to_int_list)
        .def("multiply_scalar", [](const TensorDataPtr& a, float s) { return get_backend(a)->multiply_scalar(a, s); })
        .def("add_scalar", [](const TensorDataPtr& a, float s) { return get_backend(a)->add_scalar(a, s); })
        .def("divide_scalar", [](const TensorDataPtr& a, float s) { return get_backend(a)->divide_scalar(a, s); })
        .def("__repr__", &TensorData::to_string);

    // Math Kernels using Dispatcher
    m.def("add", [](const TensorDataPtr& a, const TensorDataPtr& b) { return get_backend(a)->add(a, b); });
    m.def("add_scalar", [](const TensorDataPtr& a, float s) { return get_backend(a)->add_scalar(a, s); });
    m.def("sub", [](const TensorDataPtr& a, const TensorDataPtr& b) { return get_backend(a)->sub(a, b); });
    m.def("divide", [](const TensorDataPtr& a, const TensorDataPtr& b) { return get_backend(a)->divide(a, b); });
    m.def("divide_scalar", [](const TensorDataPtr& a, float s) { return get_backend(a)->divide_scalar(a, s); });
    m.def("multiply", [](const TensorDataPtr& a, const TensorDataPtr& b) { return get_backend(a)->multiply(a, b); });
    m.def("multiply_scalar", [](const TensorDataPtr& a, float s) { return get_backend(a)->multiply_scalar(a, s); });
    
    m.def("sqrt", [](const TensorDataPtr& a) { return get_backend(a)->sqrt(a); });
    m.def("sum", [](const TensorDataPtr& a) { return get_backend(a)->sum(a); });
    

    m.def("embedding_forward", [](const TensorDataPtr& i, const TensorDataPtr& w) { return get_backend(w)->embedding_forward(i, w); });
    m.def("masked_fill", [](const TensorDataPtr& a, const TensorDataPtr& m, float v) { return get_backend(a)->masked_fill(a, m, v); });
    m.def("tril", [](const std::vector<size_t>& s, Device d) { return Dispatcher::instance().get_backend(d)->tril(s); }, 
          py::arg("shape"), py::arg("device") = Device::CPU);
    m.def("narrow", [](const TensorDataPtr& a, size_t d, size_t s, size_t l) { return get_backend(a)->narrow(a, d, s, l); });
    m.def("gemm", [](const TensorDataPtr& a, const TensorDataPtr& b) { return get_backend(a)->gemm(a, b); });
    
    m.def("from_list_int32", [](const std::vector<int32_t>& list, const std::vector<size_t>& shape) {
        auto data = std::make_shared<TensorData>(shape, DType::Int32, Device::CPU);
        std::memcpy(data->data(), list.data(), list.size() * sizeof(int32_t));
        return data;
    });

    m.def("softmax_cross_entropy", [](const TensorDataPtr& l, const TensorDataPtr& t) { return get_backend(l)->softmax_cross_entropy(l, t); });
    m.def("softmax_cross_entropy_backward", [](const TensorDataPtr& g, const TensorDataPtr& l, const TensorDataPtr& t) { 
        return get_backend(l)->softmax_cross_entropy_backward(g, l, t); 
    });
    
    m.def("relu", [](const TensorDataPtr& a) { return get_backend(a)->relu(a); });
    m.def("softmax", [](const TensorDataPtr& a) { return get_backend(a)->softmax(a); });
    m.def("layernorm", [](const TensorDataPtr& a, const TensorDataPtr& w, const TensorDataPtr& b, float e) { 
        return get_backend(a)->layernorm(a, w, b, e); 
    }, py::arg("a"), py::arg("weight") = nullptr, py::arg("bias") = nullptr, py::arg("eps") = 1e-5f);
    
    m.def("randn", [](const std::vector<size_t>& s, DType dt, Device d) { 
        return Dispatcher::instance().get_backend(d)->randn(s, dt); 
    }, py::arg("shape"), py::arg("dtype") = DType::Float32, py::arg("device") = Device::CPU);
    
    m.def("zeros", [](const std::vector<size_t>& s, DType dt, Device d) { 
        return Dispatcher::instance().get_backend(d)->zeros(s, dt); 
    }, py::arg("shape"), py::arg("dtype") = DType::Float32, py::arg("device") = Device::CPU);
    
    m.def("ones", [](const std::vector<size_t>& s, DType dt, Device d) { 
        return Dispatcher::instance().get_backend(d)->ones(s, dt); 
    }, py::arg("shape"), py::arg("dtype") = DType::Float32, py::arg("device") = Device::CPU);
}
