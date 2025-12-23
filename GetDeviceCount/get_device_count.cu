#include <iostream>
#include <cuda_runtime.h>
#include "cuda_check.h" // error-handling for the CUDA runtime API

using namespace std;


int main(){
    int count;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    cout << "The number of devices we have are " << count << endl;
    return 0;
}