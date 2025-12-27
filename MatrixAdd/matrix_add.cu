#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include "cuda_check.h"
using namespace std;
using namespace std::chrono;

//#define get_matrix_index(i, j, width)((j) + (i)*(width))
__host__ __device__ int get_mat_ind(int i, int j, int n_columns){
    return (j + i * n_columns);
}


void matrix_mul(float* A, float* B, float* C, int M, int N, int P){
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
    for (int k=0; k<N; k++){
        for (int i=0; i < M; i++){
            h_A[get_mat_ind(i, k, N)] = i + k;
        }
        for(int j=0; j < P; j++){
            h_B[get_mat_ind(k, j, P)] = 2 * (k + j);

        }
    }
    matrix_mul(h_A, h_B, h_C_CPU, M, N, P);
    print_array(h_C_CPU, M*P);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_CPU;
    return 0;
}