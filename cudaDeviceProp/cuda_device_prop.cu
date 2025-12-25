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

    cout << endl;
    
    cout << "Max grid dimensions (x, y, z): "
     << device_1_prop.maxGridSize[0] << ", "
     << device_1_prop.maxGridSize[1] << ", "
     << device_1_prop.maxGridSize[2] << endl;
     
    cout << "Max block dimensions (x, y, z): "
     << device_1_prop.maxThreadsDim[0] << ", "
     << device_1_prop.maxThreadsDim[1] << ", "
     << device_1_prop.maxThreadsDim[2] << endl;

    cout << "Max threads per block: " << device_1_prop.maxThreadsPerBlock << endl;
    cout << "Wrap size (for threads) " << device_1_prop.warpSize << endl;
    cout << endl; 
    cout << "Device Major: " << device_1_prop.major << endl;
    cout << "Device Minor: " << device_1_prop.minor << endl;
    cout << "Number of streaming Multi-processors: " << device_1_prop.multiProcessorCount;
    return 0;
}