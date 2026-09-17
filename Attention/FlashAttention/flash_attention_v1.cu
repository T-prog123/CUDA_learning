#include <iostream>
#include <cmath>
#include <limits>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math_constants.h>
#include "../../include/cuda_check.h"
#include "common.h"


__global__ void flash_attention_v1(){

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
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), N*d*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), N*d*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), N*d*sizeof(float), cudaMemcpyHostToDevice));

    
    // Part 1 on the attention algorithm: get the scores S = sqrt(Q K^T) / sqrt(d)

    // Part 2 of the attention algorithm: get the probabilites P = softmax(S)
  
    // Part 3 of the attention algorithm: get the output O = PV


    const int block_size = 32;
    dim3 blockDim(block_size);
    dim3 gridDim((N + block_size - 1) / block_size);

    flash_attention_v1<<<gridDim, blockDim>>>(d_Q, d_K, d_V, d_O, N, d);
    "O_s    // partial unnormalized output, shape [Br, d]
    m_s    // row max for that split, shape [Br]
    l_s    // row sum of exp(scores - m_s), shape [Br]" 
    // the outputs, they are partial
    fa_reduction_v1<<>>();
    // reduces the previous results do give the final output matrix
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_O.data(), d_O, N*d*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
    return 0;
}