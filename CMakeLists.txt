cmake_minimum_required(VERSION 3.18)
project(bitonic_sort LANGUAGES C CXX CUDA)

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.12.0  # or latest release
)
FetchContent_MakeAvailable(pybind11)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

add_library(bitonic_sort MODULE
    bitonic_sort.cu
    bitonic_wrapper.cpp
)

target_link_libraries(bitonic_sort PRIVATE pybind11::module)

set_target_properties(bitonic_sort PROPERTIES
    PREFIX ""
    CUDA_SEPARABLE_COMPILATION ON
)
