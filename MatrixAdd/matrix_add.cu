#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include "cuda_check.h"
using namespace std;
using namespace std::chrono;

const int block_size = 128;

//#define get_matrix_index(i, j, width)((j) + (i)*(width))
__host__ __device__ int get_mat_ind(int i, int j, int n_columns){
    return (j + i * n_columns);
}

__global__ void MatrixMul(const float* d_A, const float* d_B, float* d_C, const int M, const int N, const int P){

}
void d_matrix_mul(const float* h_A, const float* h_B, float* h_C_GPU, const int M, const int N, const int P){
    // setting the variables and the memory transfers up
    float *d_A;
    float *d_B;
    float *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M*N*sizeof(float)) );
    CUDA_CHECK(cudaMalloc(&d_B, N*P*sizeof(float)) );
    CUDA_CHECK(cudaMalloc(&d_C, M*P*sizeof(float)) );
    CUDA_CHECK(cudaMemcpy(d_A, h_A, M*N*sizeof(float), cudaMemcpyHostToDevice) );
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N*P*sizeof(float), cudaMemcpyHostToDevice) );

    int grid_size;

    CUDA_CHECK(cudaMemcpy(h_C_GPU, d_C, M*P*sizeof(float), cudaMemcpyDeviceToHost) );
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

void h_matrix_mul(float* A, float* B, float* C, int M, int N, int P){
    // A of shape (M, N)
    // B of shape (N, P)
    // logically, C of shape (M, P)
    for (int i=0; i < M; i++){
        for(int j=0; j < P; j++){
            double s = 0;
            for (int k=0; k<N; k++){
                int A_index = get_mat_ind(i, k, N);
                int B_index = get_mat_ind(k, j, P);
                s += A[A_index] * B[B_index];
            }
            int C_index = get_mat_ind(i, j, P);
            C[C_index] = (float)s;
        }
    }
}


int main(int argc, char** argv){
    // initialising the main (host-side) variables
    int M = 10;
    int N = 5;
    int P = 20;
    float* h_A = new float[M*N]; //shape (M, N)
    float* h_B = new float[N*P]; //shape (N, P)
    float* h_C_CPU = new float[M*P]; //shape (M, P)
    float* h_C_GPU = new float[M*P]; //shape (M, P)
    for (int k=0; k<N; k++){
        for (int i=0; i < M; i++){
            h_A[get_mat_ind(i, k, N)] = i + k;
        }
        for(int j=0; j < P; j++){
            h_B[get_mat_ind(k, j, P)] = 2 * (k + j);

        }
    }

    // running and timing the CPU-side computations
    auto host_start = high_resolution_clock::now();
    h_matrix_mul(h_A, h_A, h_C_CPU, M, N, P);
    auto host_end = high_resolution_clock::now();
    auto host_duration = duration_cast<milliseconds>(host_end - host_start);
    cout << "Time for cpu operations: " << host_duration.count() << " milli seconds" << endl;

    // freeing memory
    //print_array(h_C_CPU, M*P);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_CPU;
    delete[] h_C_GPU;
    return 0;
}