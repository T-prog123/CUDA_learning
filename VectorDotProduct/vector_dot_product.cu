#include <iostream>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>
#include "cuda_check.h"

__global__ void VectDotProduct(const float* A, const float* B, float* C, const int N){

}

void device_vector_dot_product(const float* A, const float* B, float* C, const int N){

}

void host_vector_dot_product(const float* A, const float* B, float* C, const int N){

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
    float *h_C_GPU = new float[N];
    float *h_C_CPU = new float[N];

    for (int i=0; i<N; i++){
        h_A[i] = i;
        h_B[i] = 2*i;
    }

    // running and timing computation on device (GPU, including data transfer time)
    auto device_start = high_resolution_clock::now();
    device_vector_dot_product(h_A, h_B, h_C_GPU, N);
    auto device_end = high_resolution_clock::now();
    auto device_duration = duration_cast<milliseconds>(d_end - d_start);

    // running and timing on host (CPU)
    auto host_start = high_resolution_clock::now();
    host_vector_dot_product(h_A, h_B, h_C_GPU, N);
    auto host_end = high_resolution_clock::now();
    auto host_duration = duration_cast<milliseconds>(d_end - d_start);

    // verification and output
    cout << "did the two computations return the same values? " << boolalpha  << equal(h_C_GPU, h_C_GPU + N, h_C_CPU) <<endl
    cout << "Time for gpu operations: " << device_duration.count() << " milli seconds" << endl;
    cout << "Time for cpu operations: " << host_duration.count() << " milli seconds" << endl;



}