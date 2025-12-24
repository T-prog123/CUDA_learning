#include <iostream>
#include <cuda_runtime.h>
#include "cuda_check.h"
using namespace std;

int main(){
    cudaDeviceProp device_1_prop;
    CUDA_CHECK(cudaGetDeviceProperties(&device_1_prop, 0));
    cout << "Device name: " << device_1_prop.name << endl;
    cout << "Max global memory (VRAM): " << device_1_prop.totalGlobalMem << endl;
    cout << "Max shared memory (per block): " << device_1_prop.sharedMemPerBlock << endl;
    cout << "Max threads per block: " << device_1_prop.maxThreadsPerBlock << endl;
    cout << "Wrap size (for threads) " << device_1_prop.wrapSize << endl;
    cout << endl;
    
    cout << "Device Major: " << device_1_prop.major << endl;
    cout << "Device Minor: " << device_1_prop.minor << endl;
    return 0;
}