#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include "include/bitonic.h"

__global__ void exchangeKernel(int *array, int size, int group_size, int distance)
{
    int idx = threadIdx.x + gridIdx.x * threadDim.x;
    int partner = idx ^ distance;
    bool sort_descending = idx & group_size;

    if ((!sort_descending && idx < partner) || (sort_descending && idx > partner))
    {

        // keep min elements
        if (array[idx] > array[partner])
        {
            swap(&array[idx], &array[partner]);
        }
        else
        { // keep max elements
            if (array[idx] < array[partner])
            {
                swap(&array[idx], &array[partner]);
            }
        }
    }
}

void bitonicSort(int **array, int size)
{

    for (int group_size = 2; group_size <= size; group_size <<= 1)
    { // group_size doubles in each reccursion
        bool sort_descending = rank & group_size;

        for (int distance = group_size >> 1; distance > 0; distance >>= 1)
        { // half distance

            // GPU PARAMETERS
            int threads_per_block = 1024; // max threads
            int blocks_per_grid = 1;
            int num_grid = (size / 1024);
            dim3 blocksPerGrid(1, num_grid); // 1 block per grid

            int partner_rank = ^distance;
            exchangeKernel<<<blocks_per_grid, threads_per_block>>>(array, size, group_size, distance);

            cudaDeviceSynchronize();
        }
    }
}

void swap(int *a, int *b)
{

    int temp;
    temp = *a;
    a = *b;
    b = *a;
}

void print(int *array, int size)
{

    for (int i = 0; i < size; i++)
    {
        printf("%2d ", array[i]);
    }
}

void evaluateResult(int *array, int size)
{

    bool is_Sorted;
    for (int i = 0; i < size - 1; i++)
    {

        if (array[i] > array[i + 1])
        {
            isSorted = false;
            break;
        }
    }
    is_Sorted = true;

    if (is_Sorted)
    {
        printf("Sorted array\n");
    }
    else
        printf("Array is not sorted!!\n");
}