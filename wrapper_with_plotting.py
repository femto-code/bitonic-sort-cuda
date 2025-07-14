import numpy as np
from bitonic_sort import sort
import time
import matplotlib.pyplot as plt
import csv
import os
import argparse
from datetime import datetime

def is_power_of_two(n):
    return (n & (n - 1) == 0) and n != 0

def benchmark(size):
    assert is_power_of_two(size), "Size must be a power of 2"

    a = np.random.randint(0, 10000, size, dtype=np.int32)
    a_copy = a.copy()

    # GPU Bitonic Sort
    start = time.time()
    sort(a_copy)
    cuda_time = time.time() - start

    # NumPy Sort
    start = time.time()
    sorted_cpu = np.sort(a)
    numpy_time = time.time() - start

    # Validate correctness
    match = np.allclose(a_copy, sorted_cpu)

    return cuda_time, numpy_time, match


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Bitonic Sort on GPU vs NumPy on CPU")
    parser.add_argument("--cluster", type=str, required=True, help="Name of the cluster to appear in the plot title")
    args = parser.parse_args()

    cluster_name = args.cluster

    sizes = [2**i for i in range(10, 28)]
    cuda_times = []
    numpy_times = []
    valid = []

    for size in sizes:
        print(f"\n--- Benchmarking size {size} ---")
        cuda_time, numpy_time, match = benchmark(size)
        print(f"CUDA Bitonic sort time: {cuda_time:.6f} s")
        print(f"NumPy sort time:        {numpy_time:.6f} s")
        print("✅ Results match" if match else "❌ Mismatch!")

        cuda_times.append(cuda_time)
        numpy_times.append(numpy_time)
        valid.append(match)

    # Write results to CSV
    with open(os.path.join("results", f"{cluster_name}_bitonic_sort_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"), "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Array Size", "CUDA Bitonic Sort Time (s)", "NumPy Sort Time (s)", "Results Match"])
        for size, cuda_time, numpy_time, match in zip(sizes, cuda_times, numpy_times, valid):
            writer.writerow([size, cuda_time, numpy_time, match])

    # Plotting results
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, numpy_times, 'o--', label='NumPy (CPU)', color='tab:blue')
    plt.plot(sizes, cuda_times, 'o--', label='Bitonic (CUDA)', color='tab:orange')
    plt.xlabel('Array Size')
    plt.ylabel('Time (seconds)')
    plt.title(f'Bitonic Sort Performance Comparison on Cluster {cluster_name}')
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("plots", f"{cluster_name}_bitonic_sort_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.svg"))
    plt.show()


    