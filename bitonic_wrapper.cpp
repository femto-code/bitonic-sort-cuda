#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "bitonic_sort.h"
#define BLOCK_SIZE 128


namespace py = pybind11;

void bitonicSort_py(py::array_t<int> arr_np) {
    auto buf = arr_np.request();
    int* ptr = static_cast<int*>(buf.ptr);
    int n = buf.size;

    if ((n & (n - 1)) != 0) throw std::runtime_error("Size must be power of 2.");
    if (n % BLOCK_SIZE != 0) throw std::runtime_error("Size must be divisible by BLOCK_SIZE.");

    bitonicSort(ptr, n);
}

PYBIND11_MODULE(bitonic_sort, m) {
    m.def("sort", &bitonicSort_py, "Run bitonic sort on a NumPy array");
}
