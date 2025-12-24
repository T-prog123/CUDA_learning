#include<cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cstdlib>
using namespace std;
using namespace std::chrono;

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

void add_vector_host(const float* h_A, const float* h_B, float* h_C, int N){
    for (int i=0; i<N; i++){
        h_C[i] = h_A[i] + h_B[i];
    }
}

void print_array(float* x, int N){
    for (int i=0; i<N; i++){
        cout << x[i] <<" ";
    }
    cout << endl;
}

int main(int argc, char** argv)
{   
    int N = (argc > 1) ? static_cast<int>(atof(argv[1])) : 1e3;

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
    //print_array(h_C2, N);
    cout << endl;
    cout << "Time for gpu operations: " << d_duration.count() << " milli seconds" << endl;
    cout << "Time for cpu operations: " << h_duration.count() << " milli seconds" << endl;

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    
}