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

const int block_dim_x = 4;
const int block_dim_y = 4;
const int shared_tile_dim = 4;

//#define get_matrix_index(i, j, width)((j) + (i)*(width))
__host__ __device__ int get_mat_ind(int i, int j, int n_columns){
    return (j + i * n_columns);
}

__global__ void MatrixMul_inner_product(const float* d_A, const float* d_B, float* d_C, const int M, const int N, const int P){
    __shared__ float tile_A[block_dim_x * shared_tile_dim];
    __shared__ float tile_B[shared_tile_dim * block_dim_y];
    float thread_value = 0;


    const int tile_a_idx = get_mat_ind(threadIdx.x, threadIdx.y, shared_tile_dim); // width = shared_tile_dim
    const int tile_b_idx = get_mat_ind(threadIdx.x, threadIdx.y, block_dim_y);    // width = block_dim_y


    const int A_index_x = threadIdx.x + blockIdx.x * blockDim.x; // constant index that does not change as the tile moves
    const int B_index_y = threadIdx.y + blockIdx.y * blockDim.y; // constant index that does not change as the tile moves
    //DPRINTF( "B_index_y %d\n", B_index_y);

    // we code this assuming there are enough blocks (the grid is big enough) for tiles to cover all of C at launch.
    int n_tile = 0; // tracks where our tiles on A and B are respectibely
    while (n_tile < (N + shared_tile_dim - 1) / shared_tile_dim){
        int A_index_y = threadIdx.y + n_tile * blockDim.y;
        int A_index = get_mat_ind(A_index_x, A_index_y, N);
        int B_index_x = threadIdx.x + n_tile * blockDim.x;
        int B_index = get_mat_ind(B_index_x, B_index_y, P);

        // assining values to the tile (in shared memory) depending on if we are in bound of the real matrices or not
        if (A_index_x < M && A_index_y < N){
            tile_A[tile_a_idx] = d_A[A_index];
            if (blockIdx.x ==0 && blockIdx.y == 0){
                DPRINTF("Tile A at global indexes %d, %d was filled with value %f\n", A_index_x, A_index_y, tile_A[tile_index]);
            }
        }
        else{
            tile_A[tile_a_idx] = 0;
        }

        if (B_index_x < N && B_index_y < P){
            tile_B[tile_b_idx] = d_B[B_index];
            if (blockIdx.x ==0 && blockIdx.y == 0){
                //DPRINTF("Tile B at global indexes %d, %d was filled with value %f\n", B_index_x, B_index_y, tile_B[tile_index]);
            }
        }
        else{
            tile_B[tile_b_idx] = 0;
        }
        __syncthreads();
        // #pragma unroll
        // for (int i=0; i<blockDim.x; i++){
        //     #pragma unroll
        //     for(int j = 0; j < blockDim.y; j++){
        //         thread_tile_C[get_mat_ind(i, j, gridDim.y)] = (tile_A[get_mat_ind(i, threadIdx.y, gridDim.y)] * 
        //         tile_B[get_mat_ind(threadIdx.y, j, gridDim.y)]);
        //     }
        // }
        for (int k=0; k<shared_tile_dim; k++){
            thread_value += tile_A[get_mat_ind(threadIdx.x, k, shared_tile_dim)] * 
            tile_B[get_mat_ind(k, threadIdx.y, block_dim_y)];
        }
        __syncthreads();
        n_tile += 1;
    }
    if (A_index_x < M && B_index_y < P){
        d_C[get_mat_ind(A_index_x, B_index_y, P)] = thread_value;
    }
    // DPRINTF("When writting at the end, we access index %d, %f we are writting value\n", get_mat_ind(A_index_x, B_index_y, P), thread_value);
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

    // preparing dimensions for kernel launch
    const int grid_dim_x = min((M + block_dim_x - 1) / block_dim_x, 10000);
    const int grid_dim_y = min((P + block_dim_y - 1) / block_dim_y, 10000);
    dim3 grid_dim(grid_dim_x, grid_dim_y, 1);
    dim3 block_dim(block_dim_x, block_dim_y, 1);

    // preparing timing for kernel launch
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0); 

    MatrixMul_inner_product<<<grid_dim, block_dim>>>(d_A, d_B, d_C, M, N, P);

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
                s += (double)A[A_index] * (double)B[B_index];
            }
            int C_index = get_mat_ind(i, j, P);
            C[C_index] = (float)s;
        }
    }
}


int main(int argc, char** argv){
    // initialising the main (host-side) variables
    int M = 100;
    int N = 5000;
    int P = 200;
    float* h_A = new float[M*N]; //shape (M, N)
    float* h_B = new float[N*P]; //shape (N, P)
    float* h_C_CPU = new float[M*P]; //shape (M, P)
    float* h_C_GPU = new float[M*P]; //shape (M, P)
    for (int k=0; k<N; k++){
        for (int i=0; i<M; i++){
            h_A[get_mat_ind(i, k, N)] = (i + k) % 20;
        }
        for(int j=0; j < P; j++){
            h_B[get_mat_ind(k, j, P)] = (2 * (k + j)) % 30;

        }
    }

    // running and timing the CPU-side computations
    auto host_start = high_resolution_clock::now();
    h_matrix_mul(h_A, h_B, h_C_CPU, M, N, P);
    auto host_end = high_resolution_clock::now();
    auto host_duration = duration_cast<milliseconds>(host_end - host_start);
    cout << "Time for cpu operations: " << host_duration.count() << " milli seconds" << endl;

    // running and timing the GPU-side computations
    auto device_start = high_resolution_clock::now();
    d_matrix_mul(h_A, h_B, h_C_GPU, M, N, P);
    auto device_end = high_resolution_clock::now();
    auto device_duration = duration_cast<milliseconds>(device_end - device_start);
    cout << "Time for GPU operations: " << device_duration.count() << " milli seconds" << endl;
    cout << "did the two computations return the same values? " << boolalpha  << equal(h_C_CPU, h_C_CPU + M*P, h_C_GPU) <<endl;
    // print_array(h_C_CPU, M*P);
    // print_array(h_C_GPU, M*P);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_CPU;
    delete[] h_C_GPU;
    return 0;
}