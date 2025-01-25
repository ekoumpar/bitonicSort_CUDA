#include <stdbool.h>
#include <stdio.h>
#include <cuda.h>

#include "bitonic.h"

__device__ void swap(int *array, int idx, int partner)
{
    int temp;
    temp = array[idx];
    array[idx] = array[partner];
    array[partner] = temp;
}

__global__ void exchange_V0(int *array, int size, int group_size, int distance)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int idx = (tid / distance) * distance * 2 + (tid % distance);
    int partner = idx ^ distance;
    bool sort_descending = idx & group_size;

    if (idx < size && partner < size && idx < partner)
    { // ensure bounds are checked before accessing array

        if (!sort_descending && array[idx] > array[partner])
        {
            // keep min elements
            swap(array, idx, partner);
        }
        if (sort_descending && array[idx] < array[partner])
        {
            // keep max elements
            swap(array, idx, partner);
        }
    }
}

void bitonicSort(int *array, int size)
{
    // GPU PARAMETERS
    int threads_per_block = 1024;   //max threads per block
    int blocks_per_grid = size / threads_per_block; 

    for (int group_size = 2; group_size <= size; group_size <<= 1)
    { 
        for (int distance = group_size >> 1; distance > 0; distance >>= 1)
            exchange_V0<<<blocks_per_grid, threads_per_block>>>(array, size, group_size, distance);
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
            break;
        }
    }

    if (is_Sorted)
    {
        printf("Sorted array\n");
    }
    else
        printf("Array is not sorted!!\n");
}