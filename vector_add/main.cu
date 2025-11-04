#include<cuda_runtime.h>
#include <iostream>
using namespace std;

// Kernel definition
__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main()
{   
    int N = 100;
    // declare the host and device pointers
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];
    float *d_A, *d_B, *d_C;

    // fill in the device variables:
    for (int i=0; i<N; i++){
        h_A[i] = i;
        h_B[i] = 2 * i;
    }

    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, N * sizeof(float), cudaMemcpyHostToDevice);


    VecAdd<<<1, N>>>(d_A, d_B, d_C);

    cout <<"the CUDA code ran!" << endl;
}