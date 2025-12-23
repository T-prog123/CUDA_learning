#include <iostream>
#include <cuda_runtime.h>

using namespace std;


#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error %s:%d: %s (%s)\n",                          \
              __FILE__, __LINE__, cudaGetErrorString(err),                   \
              cudaGetErrorName(err));                                        \
      abort();                                                               \
    }                                                                        \
  } while (0)

int main(){
    int count;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    cout << "The number of devices we have are " << count << endl;
    return 0;
}