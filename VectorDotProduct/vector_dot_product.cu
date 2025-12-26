#include <iostream>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>
#include "cuda_check.h"

__global__ void VectDotProduct(const float* A, const float* B, float d_dot_product_array, const int N, const int block_size){
    int i = threadIdx.x + gridIdx.x * gridDim.x;
    __shared__ cache[block_size];

    float sum_value = 0:
    while (i < N){
        sum_value += A[i] * B[i];
        i +=  gridDim.x * blockDim.x 

    }

    cache[i] = sum_value;
    __syncthreads();

    float reduction = 0;
    if (i % (gridDim.x * blockDim.x ) == 0){
        for (int j = 0; j < block_size; j++){
            reduction += cache[j];
        }
    }
    d_dot_product_array[gridIdx.x] = reduction;
}

float device_vector_dot_product(const float* h_A, const float* h_B, const int N){
    // defining the variables needed to performe GPU-computations
    float dot_product_value = 0;
    float* d_A;
    float* d_B;
    float* h_dot_product_array;
    float* d_dot_product_array;

    // allocate and copy memory 
    CUDA_CHECK(cudaMalloc(&d_A, N * sizeof(float)) );
    CUDA_CHECK(cudaMalloc(&d_B, N * sizeof(float)) );
    CUDA_CHECK(cudaMemccpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice) );
    CUDA_CHECK(cudaMemccpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice) );

    // setting the dimensions and launching the kernel
    int block_size = 128;
    int gird_size = min(96, (N + block_size - 1) / block_size);
    dim3 block_dim(block_size, 1, 1);
    dim3 grid_dim(grid_size, 1, 1);
    CUDA_CHECK(cudaMalloc(&d_dot_product_array, gird_size * sizeof(float)) );
    VectDotProduct<<<grid_dim, block_dim>>>(d_A, d_B, d_dot_product_array, N, block_size);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaMemccpy(h_dot_product_array, d_dot_product_array, grid_size * sizeof(float)) );
    // Free memory and return value
    for (const float& x : h_dot_product_array) {
        dot_product_value += x;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_dot_product_array)
    return(dot_product_value);
}

float host_vector_dot_product(const float* A, const float* B, const int N){
    float dot_product_value = 0;
    for (int i=0; i <N; i++){
        dot_product_value += A[i] * B[i];
    }
    return dot_product_value
}

int main(){
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
    auto device_duration = duration_cast<milliseconds>(d_end - d_start);

    // running and timing on host (CPU)
    auto host_start = high_resolution_clock::now();
    h_dot_product_CPU = host_vector_dot_product(h_A, h_B, N);
    auto host_end = high_resolution_clock::now();
    auto host_duration = duration_cast<milliseconds>(d_end - d_start);

    // verification and output
    cout << "did the two computations return the same values? " << boolalpha  << equal(h_C_GPU, h_C_GPU + N, h_C_CPU) <<endl
    cout << "Time for gpu operations: " << device_duration.count() << " milli seconds" << endl;
    cout << "Time for cpu operations: " << host_duration.count() << " milli seconds" << endl;


    delete[] h_A;
    delete[] h_B;
}