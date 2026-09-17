#include <iostream>
#include <cuda_runtime.h>
#include "../include/cuda_check.h"




int main(){
    // The dimensions we will work with
    const int N = 2048;
    const int d = 128;
    
    // Initalize the variables
    auto [h_Q, h_K, h_V, h_O] = matrix_init(N, d);

    // Allocate device-side memory
    CUDA_CHECK(cudaMalloc(&d_Q, N*d*sizeof(float)) );
    CUDA_CHECK(cudaMalloc(&d_K, N*d*sizeof(float)) );
    CUDA_CHECK(cudaMalloc(&d_V, N*d*sizeof(float)) );
    CUDA_CHECK(cudaMalloc(&d_O, N*d*sizeof(float)) );
    // H2D transfer
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, N*d*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, N*d*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, N*d*sizeof(float), cudaMemcpyHostToDevice));

    attention_v0(d_Q, d_K, d_V, d_O);

    CUDA_CHECK(cudaMemcpy(h_O, d_O, N*d*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
    return 0;
}