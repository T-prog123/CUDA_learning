#include <iostream>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include "cuda_check.h"
using namespace std;
using namespace std::chrono;

const int block_size = 128;

__global__ void VectDotProduct(const float* A, const float* B, float* d_dot_product_array, const int N){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float cache[block_size];

    float sum_value = 0;
    while (i < N){
        printf("We are computing the product at %i\n", i);
        sum_value += A[i] * B[i];
        i +=  gridDim.x * blockDim.x;

    }

    cache[threadIdx.x] = sum_value;
    __syncthreads();

    float reduction = 0;
    if (threadIdx.x == 0){
        printf("We are at block %i\n", i);
        for (int j = 0; j < block_size; j++){
            if ( j + blockIdx.x * blockDim.x < N){
                printf("We are adding element %i to the reduction\n", j);
                reduction += cache[j];
            }
        }
        printf("Accessing the reduction array at %i\n", blockIdx.x);
        d_dot_product_array[blockIdx.x] = reduction;
    }
}

float device_vector_dot_product(const float* h_A, const float* h_B, const int N){
    // defining the variables needed to performe GPU-computations
    float dot_product_value = 0;
    float* d_A;
    float* d_B;

    // allocate and copy memory 
    CUDA_CHECK(cudaMalloc(&d_A, N * sizeof(float)) );
    CUDA_CHECK(cudaMalloc(&d_B, N * sizeof(float)) );
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice) );
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice) );

    // setting the dimensions and launching the kernel
    const int grid_size = min(96, (N + block_size - 1) / block_size);
    dim3 block_dim(block_size, 1, 1);
    dim3 grid_dim(grid_size, 1, 1);
    float* h_dot_product_array = new float[grid_size];
    float* d_dot_product_array;
    CUDA_CHECK(cudaMalloc(&d_dot_product_array, grid_size * sizeof(float)) );
    VectDotProduct<<<grid_dim, block_dim>>>(d_A, d_B, d_dot_product_array, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_dot_product_array, d_dot_product_array, grid_size * sizeof(float), cudaMemcpyDeviceToHost) );
    // Free memory and return value
    for (int i=0; i < grid_size; i++) {
        dot_product_value += h_dot_product_array[i];
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_dot_product_array);
    delete h_dot_product_array;
    return(dot_product_value);
}

float host_vector_dot_product(const float* A, const float* B, const int N){
    float dot_product_value = 0;
    for (int i=0; i <N; i++){
        dot_product_value += A[i] * B[i];
    }
    return dot_product_value;
}

int main(int argc, char** argv){
    // setting things up:
    int N;
    if (argc > 1){
        N = static_cast<int>(atof(argv[1]));
    }
    else {
        N = 1000;
    }

    float *h_A = new float[N];
    float *h_B = new float[N];
    float h_dot_product_GPU;
    float h_dot_product_CPU;

    for (int i=0; i<N; i++){
        h_A[i] = i;
        h_B[i] = 2*i;
    }

    // running and timing computation on device (GPU, including data transfer time)
    auto device_start = high_resolution_clock::now();
    h_dot_product_GPU = device_vector_dot_product(h_A, h_B, N);
    auto device_end = high_resolution_clock::now();
    auto device_duration = duration_cast<milliseconds>(device_end - device_start);

    // running and timing on host (CPU)
    auto host_start = high_resolution_clock::now();
    h_dot_product_CPU = host_vector_dot_product(h_A, h_B, N);
    auto host_end = high_resolution_clock::now();
    auto host_duration = duration_cast<milliseconds>(host_end - host_start);

    // verification and output

    bool equal = std::abs(h_dot_product_GPU - h_dot_product_CPU) < 1e-6f;
    cout << endl;
    cout << "GPU computed values: " << h_dot_product_GPU << endl;
    cout << "CPU computed values: " << h_dot_product_CPU << endl;
    cout << "did the two computations return the same values? "
        << boolalpha
        << equal
        << endl;

    cout << "Time for gpu operations: " << device_duration.count() << " milli seconds" << endl;
    cout << "Time for cpu operations: " << host_duration.count() << " milli seconds" << endl;


    delete[] h_A;
    delete[] h_B;
}