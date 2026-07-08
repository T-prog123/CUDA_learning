#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <mma.h>
#include "cuda_check.h"
#include <algorithm>
using namespace std;
using namespace std::chrono;
using namespace nvcuda;
#include <cassert>


// for debuging: 
# define DEBUG 0

#if DEBUG
  #define DPRINTF(...) do { printf(__VA_ARGS__); } while (0)
#else
  #define DPRINTF(...) do {} while (0)
#endif

const int block_dim_x = 32;
const int block_dim_y = 16;
const int warp_tile_m = 16;
const int warp_tile_n = 16;
const int warp_tile_k = 8;
const int warps_per_block_x = 4;
const int warps_per_block_y = 4;
const int output_tile_dim_x = warps_per_block_x * warp_tile_n;
const int output_tile_dim_y = warps_per_block_y * warp_tile_m;

//#define get_matrix_index(i, j, width)((j) + (i)*(width))
__host__ __device__ int get_mat_ind(int i, int j, int n_columns){
    return (j + i * n_columns);
}

__global__ void MatrixMul_tensor_cores(const float* d_A, const float* d_B, float* d_C, const int M, const int N, const int P){
    const int warp_id = threadIdx.y;
    const int warp_row = warp_id / warps_per_block_x;
    const int warp_col = warp_id % warps_per_block_x;

    const int C_row = blockIdx.y * output_tile_dim_y + warp_row * warp_tile_m;
    const int C_col = blockIdx.x * output_tile_dim_x + warp_col * warp_tile_n;

    wmma::fragment<wmma::matrix_a, warp_tile_m, warp_tile_n, warp_tile_k, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, warp_tile_m, warp_tile_n, warp_tile_k, wmma::precision::tf32, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, warp_tile_m, warp_tile_n, warp_tile_k, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < N; k += warp_tile_k){
        wmma::load_matrix_sync(a_frag, d_A + get_mat_ind(C_row, k, N), N);
        wmma::load_matrix_sync(b_frag, d_B + get_mat_ind(k, C_col, P), P);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    if (C_row + warp_tile_m <= M && C_col + warp_tile_n <= P){
        wmma::store_matrix_sync(d_C + get_mat_ind(C_row, C_col, P), c_frag, P, wmma::mem_row_major);
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
    const int grid_dim_x = (P + output_tile_dim_x - 1) / output_tile_dim_x;
    const int grid_dim_y = (M + output_tile_dim_y - 1) / output_tile_dim_y;
    dim3 grid_dim(grid_dim_x, grid_dim_y, 1);
    dim3 block_dim(block_dim_x, block_dim_y, 1);

    // preparing timing for kernel launch
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0); 

    MatrixMul_tensor_cores<<<grid_dim, block_dim>>>(d_A, d_B, d_C, M, N, P);

    // timing and memory-check for kernel launch
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);         // wait for kernel completion
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel time: %.3f ms\n", ms);
    double gflops = (2.0 * (double)M * (double)N * (double)P) / (ms * 1.0e6);
    printf("Achieved GFLOP/s: %.2f\n", gflops);
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
