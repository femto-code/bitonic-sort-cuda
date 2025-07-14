#!/bin/bash

#SBATCH --job-name=aca-bitonic-sort
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=exercise-gpu
#SBATCH --gres=gpu:1

spack env activate cuda
spack load cuda@12.1.1

rm -rf build
mkdir build && cd build
cmake ..
make
cp bitonic_sort.so ../
cd ..
srun python3 wrapper_with_plotting.py --cluster brook