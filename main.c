#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#include "include/bitonic.h"

int main() {
    int q = 10;
    size_t size = 1 << q ; // 2^q elements
    
    double start_time, end_time;

    // Memory allocation
    int *GPU_array;       
    int *CPU_array;            

    cudaMalloc(&GPU_array, size * sizeof(int));
    CPU_array = (int*)malloc(size * sizeof(int));

    // Initiallize CPU array and send to GPU

    for (int i = 0; i < size; i++) {
        CPU_array[i] = rand() % 1000; // random numbers from (0-1000)
    }

    // Send array to GPU
    cudaMemcpy(GPU_array, CPU_array, size * sizeof(int), cudaMemcpyHostToDevice);

    start_time = MPI_Wtime();

    // Bitonic sort in GPU
    bitonicSort(GPU_array, size);

    
    end_time = MPI_Wtime();

    // Send array back to CPU
    cudaMemcpy(GPU_array, CPU_array, size * sizeof(int), cudaMemcpyDeviceToHost);


    //print(CPU_array, size);

    printf("Execution time: %f seconds\n", end_time - start_time);
    
    evaluateResult(CPU_array, size);

    cudaFree(GPU_array);
    free(CPU_array);
    
    return 0;
}