#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "bitonic.h"

int main() {
    int q = 10;
    size_t size = 1 << q ; // 2^q elements
    
    // Time variables
    cudaEvent_t start, end;
    float exe_time;

    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // Memory allocation
    int *GPU_array;                  
    cudaError_t err = cudaMallocManaged(&GPU_array,  size * sizeof(int));

    if (err != cudaSuccess) {
      printf("Error by Memory allocation\n \"%s\"", cudaGetErrorString(err));
      return -1;
    }
    
    srand(time(0));
    for (int i = 0; i < size; i++) {
      GPU_array[i] = rand() % 1000; // random numbers from (0-1000)
    }

    //print(GPU_array, size);
    cudaEventRecord(start);

    // Bitonic sort in GPU
    bitonicSort(GPU_array, size);

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&exe_time, start, end);

    //print(GPU_array, size);

    evaluateResult(GPU_array, size);
    printf("Execution time: %f ms", exe_time);
  
    cudaFree(GPU_array);
    
    return 0;
}