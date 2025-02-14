#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "bitonic.h"

int main(int argc, char *argv[])
{
  int q = atoi(argv[1]);
  size_t size = 1 << q; // 2^q elements

  // Time variables
  cudaEvent_t start, end;
  float exe_time;

  cudaEventCreate(&start);
  cudaEventCreate(&end);

  // Memory allocation
  int *CPU_array = (int *)malloc(size * sizeof(int));
  int *GPU_array;

  cudaError_t err = cudaMalloc(&GPU_array, size * sizeof(int));
  if (err != cudaSuccess)
  {
    printf("Error by Memory allocation\n \"%s\"", cudaGetErrorString(err));
    return -1;
  }

  srand(time(0));
  for (int i = 0; i < size; i++)
  {
    CPU_array[i] = rand() % 1000; // random numbers from (0-1000)
  }

  // print(GPU_array, size);
  cudaEventRecord(start);

  cudaMemcpy(GPU_array, CPU_array, size * sizeof(int), cudaMemcpyHostToDevice);

  // Bitonic sort in GPU
  bitonicSort(GPU_array, size);

  cudaMemcpy(CPU_array, GPU_array, size * sizeof(int), cudaMemcpyDeviceToHost);

  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&exe_time, start, end);

  // print(GPU_array, size);

  evaluateResult(CPU_array, size);
  printf("Execution time: %f s", exe_time / 1000);

  cudaEventDestroy(start);
  cudaEventDestroy(end);

  cudaFree(GPU_array);
  free(CPU_array);

  return 0;
}