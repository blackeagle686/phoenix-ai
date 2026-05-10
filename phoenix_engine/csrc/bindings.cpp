#include <pybind11/pybind11.h>
#include <string>

namespace py = pybind11;

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
}
