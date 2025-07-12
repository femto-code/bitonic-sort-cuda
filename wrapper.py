# import numpy as np
# import bitonic_sort

# data = np.random.randint(0, 1000, 1024).astype(np.int32)
# print(data[:10])
# bitonic_sort.sort(data)
# print(data[:10])

import numpy as np
# import sm_version.build.bitonic_sort
import bitonic_sort
import time


def is_power_of_two(n):
    return (n & (n - 1) == 0) and n != 0

def benchmark(size):
    assert is_power_of_two(size), "Size must be a power of 2"

    a = np.random.randint(0, 10000, size, dtype=np.int32)
    a_copy = a.copy()

    print(a_copy[:10])

    start = time.time()
    # Bitonic sort on GPU operates in place
    bitonic_sort.sort(a_copy)
    end = time.time()
    print(f"CUDA Bitonic sort time: {end - start:.6f} s")

    print(a_copy[:10])

    start = time.time()
    sorted_cpu = np.sort(a)
    end = time.time()
    print(f"NumPy sort time:       {end - start:.6f} s")

    print(sorted_cpu[:10])

    if np.allclose(a_copy, sorted_cpu):
        print("✅ Results match with NumPy")
    else:
        print("❌ Results do NOT match!")

if __name__ == "__main__":
    for size in [2**9, 2**10, 2**14, 2**18]:  # Test increasing sizes
        print(f"\n--- Benchmarking size {size} ---")
        benchmark(size)
