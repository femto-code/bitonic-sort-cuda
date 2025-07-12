# Bitonic Sort (GPU CUDA optimization)

This project implements a highly optimized bitonic sort algorithm using CUDA for GPU acceleration. The implementation includes both a standalone C++ version and a Python wrapper for benchmarking against NumPy's CPU-based sort.

## Overview

Bitonic sort is a comparison-based sorting algorithm that works particularly well on parallel architectures like GPUs. This implementation uses a hybrid approach:

1. **Block-level sorting**: Each thread block sorts data in shared memory
2. **Global merging**: Cross-block bitonic merge operations for the complete sort

## Implementation

### Core Components

- **`bitonic_sort.cu`**: Main CUDA implementation with kernels
- **`bitonic_wrapper.cpp`**: PyBind11 Python wrapper
- **`wrapper_with_plotting.py`**: Python benchmarking script with visualization

### Algorithm Structure

1. **Input Validation**: Ensures array size is a power of 2 and divisible by block size
2. **Memory Transfer**: Copies data from host to device memory
3. **Block-wise Sort**: Each 128-thread block sorts its portion using shared memory
4. **Global Merge**: Iterative bitonic merge operations across the entire array
5. **Result Transfer**: Copies sorted data back to host

### CUDA Kernels

- **`bitonicSortShared`**: Sorts data within each thread block using shared memory
- **`bitonicMergeKernel`**: Performs bitonic merge operations across blocks

## Building the Project

The project uses CMake for building. The build process is automated in the cluster scripts:

```bash
rm -rf build
mkdir build && cd build
cmake ..
make
cp bitonic_sort.so ../
```

## Running on Clusters

### Moore Cluster

**GPU Architecture Details**:
- NVIDIA GeForce RTX 2080 Ti
- SM count: 68
- Shared memory per block: 49152

To run the benchmark on the Moore cluster:

```bash
sbatch run_moore.sh
```

This script:
- Uses the `exercise-gpu` partition
- Allocates 1 GPU
- Loads CUDA 12.1.1 via Spack
- Builds the project and runs the benchmark
- Saves results with "moore" cluster identifier

### Brook Cluster

**GPU Architecture Details**:
- NVIDIA TITAN X (Pascal)
- SM count: 28
- Shared memory per block: 49152

To run the benchmark on the Brook cluster:

```bash
sbatch run_brook.sh
```

This script:
- Uses the `exercise-hpdc` partition
- Allocates 1 GPU
- Loads CUDA 12.1.1 via Spack
- Builds the project and runs the benchmark
- Saves results with "brook" cluster identifier

## Results

- The resulting plots are found in the `plots` directory.
- The results with array size and whether the results are correct are exported as `.csv` in `results`
