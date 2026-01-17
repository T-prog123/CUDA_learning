#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error %s:%d: %s (%s)\n",                         \
              __FILE__, __LINE__, cudaGetErrorString(err),                   \
              cudaGetErrorName(err));                                        \
      abort();                                                               \
    }                                                                        \
  } while (0)

void print_array(const float* x, const int N){
    for (int i=0; i<N; i++){
        std::cout << x[i] <<" ";
    }
    std::cout << std::endl;
}