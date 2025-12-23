#include <iostream>
int main{
    int count;
    cudaGetDeviceCount(count);
    cout << "The number of devices we have are" << count << endl;
    return 0
}