#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <algorithm>
#include <cassert>
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


void add_vector_device(const float* h_A, const float *h_B, float *h_C, const int N, 
    cudaStream_t stream_1, cudaStream_t stream_2, const int stream_size){
    const int block_size = 128;
    const int grid_size = min(96, (N + block_size - 1)/block_size); 
    cout << "block_size: " << block_size << endl;
    cout << "grid_size: " << grid_size << endl;
    float* d_A1;
    float* d_B1;
    float* d_C1;
    float* d_A2;
    float* d_B2;
    float* d_C2;

    size_t size =  stream_size * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_A1, size));
    CUDA_CHECK(cudaMalloc(&d_B1, size));
    CUDA_CHECK(cudaMalloc(&d_C1, size));
    CUDA_CHECK(cudaMalloc(&d_A2, size));
    CUDA_CHECK(cudaMalloc(&d_B2, size));
    CUDA_CHECK(cudaMalloc(&d_C2, size));

    // lauching the host-to-device, kernel, and device-to-host asynchronously with 2 streams
    for (int i = 0; i < N; i += 2 * stream_size){    
        // copying for host to device (engine 1)
        CUDA_CHECK(cudaMemcpyAsync(d_A1, h_A + i, size, cudaMemcpyHostToDevice, stream_1));
        CUDA_CHECK(cudaMemcpyAsync(d_B1, h_B + i, size, cudaMemcpyHostToDevice, stream_1));
        CUDA_CHECK(cudaMemcpyAsync(d_A2, h_A + i + stream_size, size, cudaMemcpyHostToDevice, stream_2));
        CUDA_CHECK(cudaMemcpyAsync(d_B2, h_B + i + stream_size, size, cudaMemcpyHostToDevice, stream_2));

        // launching the kernel (engine 2):
        dim3 grid_dimension(grid_size, 1, 1);
        dim3 block_dimension(block_size, 1, 1);
        VecAdd<<<grid_dimension, block_dimension, 0, stream_1>>> (d_A1, d_B1, d_C1, stream_size);
        VecAdd<<<grid_dimension, block_dimension, 0, stream_2>>> (d_A2, d_B2, d_C2, stream_size);
        CUDA_CHECK(cudaGetLastError());

        // copying from device to host (engine 3)
        CUDA_CHECK(cudaMemcpyAsync(h_C + i, d_C1, size, cudaMemcpyDeviceToHost, stream_1));
        CUDA_CHECK(cudaMemcpyAsync(h_C + i + stream_size, d_C2, size, cudaMemcpyDeviceToHost, stream_2));
    }
    CUDA_CHECK(cudaDeviceSynchronize());



    CUDA_CHECK(cudaFree(d_A1));
    CUDA_CHECK(cudaFree(d_B1));
    CUDA_CHECK(cudaFree(d_C1));
    CUDA_CHECK(cudaFree(d_A2));
    CUDA_CHECK(cudaFree(d_B2));
    CUDA_CHECK(cudaFree(d_C2));

}

void add_vector_host(const float* h_A, const float* h_B, float* h_C, const int N){
    for (int i=0; i<N; i++){
        h_C[i] = h_A[i] + h_B[i];
    }
}

int main(int argc, char** argv)
{   
    int N;
    if (argc > 1) {
        N = static_cast<int>(atof(argv[1]));
    } else {
        N = 100000000; 
    }
    const int stream_size = N / 10;
    assert(N % stream_size == 0);

    // declare the host pointers and reserving memory
    float *h_A;
    float *h_B;
    float *h_C;
    float *h_C2;
    CUDA_CHECK(cudaMallocHost(&h_A, N * sizeof(float)) );
    CUDA_CHECK(cudaMallocHost(&h_B, N * sizeof(float)) );
    CUDA_CHECK(cudaMallocHost(&h_C, N * sizeof(float)) );
    CUDA_CHECK(cudaMallocHost(&h_C2, N * sizeof(float)) );

    cudaStream_t stream_1, stream_2;
    CUDA_CHECK(cudaStreamCreate(&stream_1));
    CUDA_CHECK(cudaStreamCreate(&stream_2));

    // fill in the device variables:
    for (int i=0; i<N; i++){
        h_A[i] = float(i);
        h_B[i] = float(2 * i);
    }

    auto d_start = high_resolution_clock::now();
    add_vector_device(h_A, h_B, h_C, N, stream_1, stream_2, stream_size);
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


    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_B));
    CUDA_CHECK(cudaFreeHost(h_C));
    CUDA_CHECK(cudaFreeHost(h_C2));
    CUDA_CHECK(cudaStreamDestroy(stream_1));
    CUDA_CHECK(cudaStreamDestroy(stream_2));
}