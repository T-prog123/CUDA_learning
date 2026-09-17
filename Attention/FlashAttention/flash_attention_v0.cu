#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../include/cuda_check.h"


void attention_v0(const float* d_Q, const float* d_K,const float* d_V, float* d_O, const int N, const int d){
    // Part 1 on the attention algorithm: get the scores S = sqrt(Q K^T) / sqrt(d)
    float* d_S;
    float* d_P;
    CUDA_CHECK(cudaMalloc(&d_S, size_t(N)*N*sizeof(float)) ); // size_t for saftey reasons, to avoid 32-bit int overflow before allocation
    CUDA_CHECK(cudaMalloc(&d_P, size_t(N)*N*sizeof(float)) );

    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f / std::sqrtf(float(d));
    float beta  = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
    N, N, d, &alpha, d_K, d, d_Q, d, &beta, d_S, N);

    // Part 2 of the attention algorithm: get the probabilites P = softmax(S)

    // Part 3 of the attention algorithm: get the output O = PV
    float one = 1.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
    d, N, N, &one, d_V, d, d_P, N, &beta, d_O, d);


    CUDA_CHECK(cudaFree(d_S));
    CUDA_CHECK(cudaFree(d_P));
    cublasDestroy(handle);
}   

int main(){
    // The dimensions we will work with
    const int N = 2048;
    const int d = 128;
    
    // Initalize the variables
    auto [h_Q, h_K, h_V, h_O] = matrix_init(N, d);

    // Allocate device-side memory
    float* d_Q;
    float* d_K;
    float* d_V;
    float* d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, N*d*sizeof(float)) );
    CUDA_CHECK(cudaMalloc(&d_K, N*d*sizeof(float)) );
    CUDA_CHECK(cudaMalloc(&d_V, N*d*sizeof(float)) );
    CUDA_CHECK(cudaMalloc(&d_O, N*d*sizeof(float)) );
    // H2D transfer
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, N*d*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, N*d*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, N*d*sizeof(float), cudaMemcpyHostToDevice));

    attention_v0(d_Q, d_K, d_V, d_O, N, d);

    CUDA_CHECK(cudaMemcpy(h_O, d_O, N*d*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
    return 0;
}