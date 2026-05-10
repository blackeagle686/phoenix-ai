#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include "core/tensor_data.h"
#include "cpu/cpu_kernels.h"

namespace py = pybind11;
using namespace phoenix;

std::string hello() {
    return "Hello from Phoenix C++ Engine!";
}

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(_phoenix_backend, m) {
    m.doc() = "Phoenix Engine Low-Level C++ Backend";
    m.def("hello", &hello, "A function that returns a hello string");
    m.def("add", &add, "A function that adds two numbers");

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
        .def("multiply_scalar", &cpu::multiply_scalar)
        .def("add_scalar", &cpu::add_scalar)
        .def("divide_scalar", &cpu::divide_scalar)
        .def("sqrt", &cpu::sqrt)
        .def("__repr__", &TensorData::to_string);

    // Math Kernels
    m.def("add", &cpu::add, "Element-wise addition of two TensorData objects");
    m.def("add_scalar", &cpu::add_scalar, "Scalar addition of a TensorData object",
          py::arg("a"), py::arg("scalar"));
    m.def("sub", &cpu::sub, "Element-wise subtraction of two TensorData objects");
    m.def("divide", &cpu::divide, "Element-wise division of two TensorData objects");
    m.def("divide_scalar", &cpu::divide_scalar, "Scalar division of a TensorData object",
          py::arg("a"), py::arg("scalar"));
    m.def("multiply", &cpu::multiply, "Element-wise multiplication of two TensorData objects");
    m.def("multiply_scalar", &cpu::multiply_scalar, "Scalar multiplication of a TensorData object",
          py::arg("a"), py::arg("scalar"));
    m.def("sqrt", &cpu::sqrt, "Element-wise sqrt of a TensorData object");
    m.def("embedding_forward", &cpu::embedding_forward, "Embedding lookup");
    m.def("masked_fill", &cpu::masked_fill, "Masked fill");
    m.def("tril", &cpu::tril, "Create lower triangular matrix");
    m.def("gemm", &cpu::gemm, "General Matrix Multiply (2D) of two TensorData objects");
    
    m.def("from_list_int32", [](const std::vector<int32_t>& list, const std::vector<size_t>& shape) {
        auto data = std::make_shared<TensorData>(shape, DType::Int32, Device::CPU);
        std::memcpy(data->data(), list.data(), list.size() * sizeof(int32_t));
        return data;
    });
    m.def("relu", &cpu::relu, "ReLU activation of a TensorData object");
    m.def("softmax", &cpu::softmax, "Fused Softmax (along the last dimension)");
    m.def("layernorm", &cpu::layernorm, "Fused LayerNorm (along the last dimension)",
          py::arg("a"), py::arg("weight") = nullptr, py::arg("bias") = nullptr, py::arg("eps") = 1e-5f);
    m.def("embedding", &cpu::embedding, "Embedding lookup table");
    m.def("randn", &cpu::randn, "Random standard normal initialization",
          py::arg("shape"), py::arg("dtype") = DType::Float32);
    m.def("zeros", &cpu::zeros, "Zeros initialization",
          py::arg("shape"), py::arg("dtype") = DType::Float32);
    m.def("ones", &cpu::ones, "Ones initialization",
          py::arg("shape"), py::arg("dtype") = DType::Float32);
}
