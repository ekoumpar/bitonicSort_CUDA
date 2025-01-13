#include <stdbool.h>
#include <stdio.h>
#include <cuda.h>

#include "bitonic.h"

__global__ void exchangeKernel(int *array, int size, int group_size, int distance)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int partner = idx ^ distance;
    bool sort_descending = idx & group_size;
	
    if (idx >= size || partner >= size) return;
    if ((!sort_descending && idx < partner) || (sort_descending && idx > partner))
    {
        // keep min elements
        if (array[idx] > array[partner])
        {
          int temp;
	        temp = array[idx];
    	    array[idx] = array[partner];
    	    array[partner] = temp;
	}    
    }
    else
    { // keep max elements
           
	 if (array[idx] < array[partner])
         {
            int temp;
            temp = array[idx];
            array[idx] = array[partner];
            array[partner] = temp;

        }
    }
}

void bitonicSort(int *array, int size)
{
    // GPU PARAMETERS
    int threads_per_block = 1024; // max threads
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;   //more if its not divided evenly


    for (int group_size = 2; group_size <= size; group_size <<= 1)
    { // group_size doubles in each reccursion

        for (int distance = group_size >> 1; distance > 0; distance >>= 1)
        { // half distance

          exchangeKernel<<<blocks_per_grid, threads_per_block>>>(array, size, group_size, distance);
	        //debbuging
          cudaError_t err = cudaGetLastError();
          if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
          }
	        cudaDeviceSynchronize();
        }
    }
}

void print(int *array, int size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%2d ", array[i]);
    }
    printf("\n");
}

void evaluateResult(int *array, int size)
{

    bool is_Sorted = true;
    for (int i = 0; i < size - 1; i++)
    {
        if (array[i] > array[i + 1])
        {
            is_Sorted = false;
            continue;
        }
    }

    if (is_Sorted)
    {
        printf("Sorted array\n");
    }
    else
        printf("Array is not sorted!!\n");
}