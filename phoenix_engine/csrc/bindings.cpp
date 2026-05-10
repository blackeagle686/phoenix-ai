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
        .def("dtype", &TensorData::dtype)
        .def("device", &TensorData::device)
        .def("size", &TensorData::size)
        .def("num_bytes", &TensorData::num_bytes)
        .def("__repr__", &TensorData::to_string);

    // Math Kernels
    m.def("add", &cpu::add, "Element-wise addition of two TensorData objects");
    m.def("multiply", &cpu::multiply, "Element-wise multiplication of two TensorData objects");
    m.def("gemm", &cpu::gemm, "General Matrix Multiply (2D) of two TensorData objects");
}
