#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <algorithm>
#include "cuda_check.h"
using namespace std;
using namespace std::chrono;




// Kernel definition
__global__ void VecAdd(const float* A, const float* B, float* C, const int N)
{   
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    /* if (i == 0 && blockIdx.x == 0){
        printf("Block Dimension X: %i\n", blockDim.x);
        printf("Grid Dimension X: %i\n", gridDim.x);
    }*/ 
    while (i < N){
        /* if (i % (blockDim.x * gridDim.x) == 0){
            printf("%i ,", i);
        }*/ 
        C[i] = A[i] + B[i];
        i += blockDim.x * gridDim.x;
    }
}


void add_vector_device(const float* h_A, const float *h_B, float *h_C, const int N){
    const int block_size = 128;
    const int grid_size = min(96, (N + block_size - 1)/block_size);
    cout << "block_size: " << block_size << endl;
    cout << "grid_size: " << grid_size << endl;
    float* d_A;
    float* d_B;
    float* d_C;

    size_t size =  N * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // launching the krenel:
    dim3 grid_dimension(grid_size, 1, 1);
    dim3 block_dimension(block_size, 1, 1);
    VecAdd<<<grid_dimension, block_dimension>>> (d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

}

void add_vector_host(const float* h_A, const float* h_B, float* h_C, const int N){
    for (int i=0; i<N; i++){
        h_C[i] = h_A[i] + h_B[i];
    }
}

void print_array(const float* x, const int N){
    for (int i=0; i<N; i++){
        cout << x[i] <<" ";
    }
    cout << endl;
}

int main(int argc, char** argv)
{   
    int N;
    if (argc > 1) {
        N = static_cast<int>(atof(argv[1]));
    } else {
        N = 1000; 
    }

    // declare the host and device pointers
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];
    float *h_C2 = new float[N];

    // fill in the device variables:
    for (int i=0; i<N; i++){
        h_A[i] = i;
        h_B[i] = 2 * i;
    }

    auto d_start = high_resolution_clock::now();
    add_vector_device(h_A, h_B, h_C, N);
    auto d_end = high_resolution_clock::now();
    auto d_duration = duration_cast<milliseconds>(d_end - d_start);
    //print_array(h_C, N);

    auto h_start = high_resolution_clock::now();
    add_vector_host(h_A, h_B, h_C2, N);
    auto h_end = high_resolution_clock::now();
    auto h_duration = duration_cast<milliseconds>(h_end - h_start);

    cout <<endl;
    // print_array(h_C2, N);
    cout << endl;
    cout << "did the two computations return the same values? " << boolalpha  << equal(h_C, h_C + N, h_C2) <<endl;
    cout << "Time for gpu operations: " << d_duration.count() << " milli seconds" << endl;
    cout << "Time for cpu operations: " << h_duration.count() << " milli seconds" << endl;

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C2;
    
}