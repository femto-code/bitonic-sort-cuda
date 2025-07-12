#include <iostream>
#include <cstdlib>
#include <ctime>

// Declare the function defined in the CUDA file
void bitonicSort(int* h_arr, int n);

int main() {
    const int N = 1024;  // must be power of 2 and divisible by BLOCK_SIZE
    int* data = new int[N];

    std::srand(std::time(nullptr));
    for (int i = 0; i < N; ++i) {
        data[i] = std::rand() % 1000;
    }

    std::cout << "Before sort:\n";
    for (int i = 0; i < 20; ++i) std::cout << data[i] << " ";
    std::cout << "\n";

    bitonicSort(data, N);

    std::cout << "After sort:\n";
    for (int i = 0; i < 20; ++i) std::cout << data[i] << " ";
    std::cout << "\n";

    delete[] data;
    return 0;
}
