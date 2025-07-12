#!/bin/bash

spack env activate cuda
spack load cuda@12.1.1

rm -rf build
mkdir build && cd build
cmake ..
make
cp bitonic_sort.so ../
cd ..
srun -p exercise-gpu --gres=gpu:1 python3 wrapper_with_plotting.py --cluster moore