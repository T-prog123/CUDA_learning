#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include "cuda_check.h"
#include <algorithm>
using namespace std;
using namespace std::chrono;
#include <cassert>


// for debuging: 
# define DEBUG 0

#if DEBUG
  #define DPRINTF(...) do { printf(__VA_ARGS__); } while (0)
#else
  #define DPRINTF(...) do {} while (0)
#endif

const int block_dim_x = 16;
const int block_dim_y = 16;
const int shared_tile_dim = 16;
const int thread_tile_dim_x = 4;
const int thread_tile_dim_y = 4;
const int output_tile_dim_x = block_dim_x * thread_tile_dim_x;
const int output_tile_dim_y = block_dim_y * thread_tile_dim_y;

//#define get_matrix_index(i, j, width)((j) + (i)*(width))
__host__ __device__ int get_mat_ind(int i, int j, int n_columns){
    return (j + i * n_columns);
}

__global__ void MatrixMul_outter_product(const float* d_A, const float* d_B, float* d_C, const int M, const int N, const int P){
    __shared__ float tile_A[output_tile_dim_y * shared_tile_dim];
    __shared__ float tile_B[shared_tile_dim * output_tile_dim_x];
    float thread_values[thread_tile_dim_y][thread_tile_dim_x] = {0};

    const int block_output_row = blockIdx.y * output_tile_dim_y;
    const int block_output_col = blockIdx.x * output_tile_dim_x;
    const int thread_output_row = threadIdx.y * thread_tile_dim_y;
    const int thread_output_col = threadIdx.x * thread_tile_dim_x;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int n_threads = blockDim.x * blockDim.y;

    for (int n_tile = 0; n_tile < (N + shared_tile_dim - 1) / shared_tile_dim; n_tile++){
        for (int idx = tid; idx < output_tile_dim_y * shared_tile_dim; idx += n_threads){
            int tile_row = idx / shared_tile_dim;
            int tile_col = idx % shared_tile_dim;
            int A_index_x = block_output_row + tile_row;
            int A_index_y = n_tile * shared_tile_dim + tile_col;

            if (A_index_x < M && A_index_y < N){
                tile_A[idx] = d_A[get_mat_ind(A_index_x, A_index_y, N)];
            }
            else{
                tile_A[idx] = 0;
            }
        }

        for (int idx = tid; idx < shared_tile_dim * output_tile_dim_x; idx += n_threads){
            int tile_row = idx / output_tile_dim_x;
            int tile_col = idx % output_tile_dim_x;
            int B_index_x = n_tile * shared_tile_dim + tile_row;
            int B_index_y = block_output_col + tile_col;

            if (B_index_x < N && B_index_y < P){
                tile_B[idx] = d_B[get_mat_ind(B_index_x, B_index_y, P)];
            }
            else{
                tile_B[idx] = 0;
            }
        }

        __syncthreads();

        for (int k=0; k<shared_tile_dim; k++){
            float thread_A[thread_tile_dim_y];
            float thread_B[thread_tile_dim_x];

            #pragma unroll
            for (int i=0; i<thread_tile_dim_y; i++){
                thread_A[i] = tile_A[get_mat_ind(thread_output_row + i, k, shared_tile_dim)];
            }

            #pragma unroll
            for (int j=0; j<thread_tile_dim_x; j++){
                thread_B[j] = tile_B[get_mat_ind(k, thread_output_col + j, output_tile_dim_x)];
            }

            #pragma unroll
            for (int i=0; i<thread_tile_dim_y; i++){
                #pragma unroll
                for (int j=0; j<thread_tile_dim_x; j++){
                    thread_values[i][j] += thread_A[i] * thread_B[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i=0; i<thread_tile_dim_y; i++){
        int C_index_x = block_output_row + thread_output_row + i;

        #pragma unroll
        for (int j=0; j<thread_tile_dim_x; j++){
            int C_index_y = block_output_col + thread_output_col + j;

            if (C_index_x < M && C_index_y < P){
                d_C[get_mat_ind(C_index_x, C_index_y, P)] = thread_values[i][j];
            }
        }
    }
}

void d_matrix_mul(const float* h_A, const float* h_B, const int M, const int N, const int P){
    // setting the variables and the memory transfers up
    float *d_A;
    float *d_B;
    float *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M*N*sizeof(float)) );
    CUDA_CHECK(cudaMalloc(&d_B, N*P*sizeof(float)) );
    CUDA_CHECK(cudaMalloc(&d_C, M*P*sizeof(float)) );
    CUDA_CHECK(cudaMemcpy(d_A, h_A, M*N*sizeof(float), cudaMemcpyHostToDevice) );
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N*P*sizeof(float), cudaMemcpyHostToDevice) );

    // preparing dimensions for kernel launch
    const int grid_dim_x = min((P + output_tile_dim_x - 1) / output_tile_dim_x, 10000);
    const int grid_dim_y = min((M + output_tile_dim_y - 1) / output_tile_dim_y, 10000);
    dim3 grid_dim(grid_dim_x, grid_dim_y, 1);
    dim3 block_dim(block_dim_x, block_dim_y, 1);

    // preparing timing for kernel launch
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0); 

    MatrixMul_outter_product<<<grid_dim, block_dim>>>(d_A, d_B, d_C, M, N, P);

    // timing and memory-check for kernel launch
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);         // wait for kernel completion
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel time: %.3f ms\n", ms);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CUDA_CHECK(cudaDeviceSynchronize()); 

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

int main(int argc, char** argv){
    // initialising the main (host-side) variables
    int M = 4096;
    int N = 4096;
    int P = 4096;
    float* h_A = new float[M*N]; //shape (M, N)
    float* h_B = new float[N*P]; //shape (N, P)
    for (int k=0; k<N; k++){
        for (int i=0; i<M; i++){
            h_A[get_mat_ind(i, k, N)] = (i + k) % 20;
        }
        for(int j=0; j < P; j++){
            h_B[get_mat_ind(k, j, P)] = (2 * (k + j)) % 30;

        }
    }

    // running and timing the GPU-side computations
    auto device_start = high_resolution_clock::now();
    d_matrix_mul(h_A, h_B, M, N, P);
    auto device_end = high_resolution_clock::now();
    auto device_duration = duration_cast<milliseconds>(device_end - device_start);
    cout << "Time for GPU operations: " << device_duration.count() << " milli seconds" << endl;
    delete[] h_A;
    delete[] h_B;
    return 0;
}
