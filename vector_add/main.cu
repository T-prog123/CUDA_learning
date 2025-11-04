#include<cuda_runtime.h>
#include <iostream>
using namespace std;

// Kernel definition
__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}


void add_vector_device(const float* h_A, const float *h_B, float *h_C, int N){
    float* d_A;
    float* d_B;
    float* d_C;

    size_t size =  N * sizeof(float);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    VecAdd<<<1,N>>> (d_A, d_B, d_C);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}

void print_array(float* x, int N){
    for (int i=0; i<N; i++){
        cout << x[i] <<" ";
    }
    cout << endl;
}

int main()
{   
    int N = 100 + 1;
    cout << "Implementation 1" << endl;
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


    VecAdd<<<1, N>>>(d_A, d_B, d_C);

    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    print_array(h_C, N);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cout << "implementation 2" << endl;

    add_vector_device(h_A, h_B, h_C, N);
    print_array(h_C, N);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
}