#include <iostream>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>
#include "cuda_check.h"
using namespace std;
using namespace std::chrono;

const int block_dim_x = 16;
const int block_dim_y = 16;

//#define get_matrix_index(i, j, width)((j) + (i)*(width))
__host__ __device__ int get_mat_ind(int i, int j, int n_columns){
    return (j + i * n_columns);
}

__global__ void MatrixAdd(const float* d_A, const float* d_B, float* d_C, const int M, const int N){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    while (i < M){
        while (j < N){
            int access_index = get_mat_ind(i, j, N);
            d_C[access_index] = d_A[access_index] + d_B[access_index];
            j += blockDim.y * gridDim.y; 
        }
        j = threadIdx.y + blockIdx.y * blockDim.y;
        i += blockDim.x * gridDim.x;
    }

}
void d_matrix_add(const float* h_A, const float* h_B, float* h_C_GPU, const int M, const int N){
    // setting the variables and the memory transfers up
    float *d_A;
    float *d_B;
    float *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M*N*sizeof(float)) );
    CUDA_CHECK(cudaMalloc(&d_B, M*N*sizeof(float)) );
    CUDA_CHECK(cudaMalloc(&d_C, M*N*sizeof(float)) );
    CUDA_CHECK(cudaMemcpy(d_A, h_A, M*N*sizeof(float), cudaMemcpyHostToDevice) );
    CUDA_CHECK(cudaMemcpy(d_B, h_B, M*N*sizeof(float), cudaMemcpyHostToDevice) );
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    int grid_dim_x = (M + block_dim_x - 1) / block_dim_x;
    int grid_dim_y = (N + block_dim_y - 1) / block_dim_y;
    dim3 block_dim(block_dim_x, block_dim_y, 1);
    dim3 grid_dim(grid_dim_x, grid_dim_y, 1);

    cudaEventRecord(start, 0);      
    MatrixAdd<<<grid_dim, block_dim>>>(d_A, d_B, d_C, M, N);
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);         // wait for kernel completion
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel time: %.3f ms\n", ms);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CUDA_CHECK(cudaDeviceSynchronize()); 

    CUDA_CHECK(cudaMemcpy(h_C_GPU, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost) );
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

void h_matrix_add(float* A, float* B, float* C, int M, int N){
    // A of shape (M, N)
    // B of shape (M, N)
    // logically, C of shape (M, N)
    for (int i=0; i < M; i++){
        for(int j=0; j < N; j++){
            int index = get_mat_ind(i, j, N);
            C[index] = A[index] + B[index];
        }
    }
}


int main(int argc, char** argv){
    // initialising the main (host-side) variables
    int M = 10000;
    int N = 5000;
    float* h_A = new float[M*N]; //shape (M, N)
    float* h_B = new float[M*N];
    float* h_C_CPU = new float[M*N]; 
    float* h_C_GPU = new float[M*N]; 
    for (int k=0; k<N; k++){
        for (int i=0; i < M; i++){
            h_A[get_mat_ind(i, k, N)] = i + k;
            h_B[get_mat_ind(i, k, N)] = 2 * (i + k);
        }
    }

    // running and timing the CPU-side computations
    auto host_start = high_resolution_clock::now();
    h_matrix_add(h_A, h_B, h_C_CPU, M, N);
    auto host_end = high_resolution_clock::now();
    auto host_duration = duration_cast<milliseconds>(host_end - host_start);
    cout << "Time for CPU operations: " << host_duration.count() << " milli seconds" << endl;

    
    // running and timing the GPU-side computations
    auto device_start = high_resolution_clock::now();
    d_matrix_add(h_A, h_B, h_C_GPU, M, N);
    auto device_end = high_resolution_clock::now();
    auto device_duration = duration_cast<milliseconds>(device_end - device_start);
    cout << "Time for GPU operations: " << device_duration.count() << " milli seconds" << endl;

    cout << "did the two computations return the same values? " << boolalpha  << equal(h_C_CPU, h_C_CPU + M*N, h_C_GPU) <<endl;

    // freeing memory
    //print_array(h_C_CPU, M*N);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_CPU;
    delete[] h_C_GPU;
    return 0;
}