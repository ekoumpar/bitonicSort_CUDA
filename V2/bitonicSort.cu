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

__device__ void load_global_to_local(int *array, int *local_array, int size, int local_tid, int offset)
{

    if (local_tid + offset < size && local_tid + offset + blockDim.x < size)
    {
        local_array[local_tid] = array[local_tid + offset];
        local_array[local_tid + blockDim.x] = array[local_tid + offset + blockDim.x];
    }

    __syncthreads();
}

__device__ void load_local_to_global(int *array, int *local_array, int size, int local_tid, int offset)
{

    if (local_tid + offset < size && local_tid + offset + blockDim.x < size)
    {
        array[local_tid + offset] = local_array[local_tid];
        array[local_tid + offset + blockDim.x] = local_array[local_tid + blockDim.x];
    }

    __syncthreads();
}

__global__ void initialExchange(int *array, int size)
{

    int local_tid = threadIdx.x;
    int global_tid = local_tid + blockIdx.x * blockDim.x;
    int offset = blockIdx.x * blockDim.x * 2;

    __shared__ int local_array[2048];

    load_global_to_local(array, local_array, size, local_tid, offset);

    for (int group_size = 2; group_size <= 1024; group_size <<= 1)
    {
        for (int distance = group_size; distance > 0; distance >>= 1)
        {
            int global_idx = (global_tid / distance) * distance * 2 + (global_tid % distance);
            int idx = (local_tid / distance) * distance * 2 + (local_tid % distance);
            int partner = idx ^ distance;
            bool sort_descending = global_idx & group_size;

            if (idx < size && partner < size && idx < partner)
            { // ensure bounds are checked before accessing array

                if (!sort_descending && local_array[idx] > local_array[partner])
                {
                    // keep min elements
                    swap(local_array, idx, partner);
                }
                if (sort_descending && local_array[idx] < local_array[partner])
                {
                    // keep max elements
                    swap(local_array, idx, partner);
                }
            }

            __syncthreads();
        }
    }
    load_local_to_global(array, local_array, size, local_tid, global_tid);
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

__global__ void exchange_V2(int *array, int size, int group_size)
{
    int local_tid = threadIdx.x;
    int global_tid = local_tid + blockIdx.x * blockDim.x;
    int offset = blockIdx.x * blockDim.x * 2;

    __shared__ int local_array[2048];

    load_global_to_local(array, local_array, size, local_tid, offset);

    for (int distance = 1024; distance > 0; distance >>= 1)
    {
        int global_idx = (global_tid / distance) * distance * 2 + (global_tid % distance);
        int idx = (local_tid / distance) * distance * 2 + (local_tid % distance);
        int partner = idx ^ distance;
        bool sort_descending = global_idx & group_size;

        if (idx < size && partner < size && idx < partner)
        { // ensure bounds are checked before accessing array

            if (!sort_descending && local_array[idx] > local_array[partner])
            {
                // keep min elements
                swap(local_array, idx, partner);
            }
            if (sort_descending && local_array[idx] < local_array[partner])
            {
                // keep max elements
                swap(local_array, idx, partner);
            }
        }

        __syncthreads();
    }

    load_local_to_global(array, local_array, size, local_tid, global_tid);
}

__host__ void bitonicSort(int *array, int size)
{
    // GPU PARAMETERS
    int threads_per_block = 1024;
    int blocks_per_grid = size / threads_per_block;

    initialExchange<<<blocks_per_grid, threads_per_block>>>(array, size);

    for (int group_size = 2048; group_size <= size; group_size <<= 1)
    { // group_size doubles in each reccursion

        int distance = group_size >> 1;

        // Handle large distances (>1024)
        while (distance > 1024)
        {
            exchange_V0<<<blocks_per_grid, threads_per_block>>>(array, size, group_size, distance);

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
                printf("CUDA Error: %s\n", cudaGetErrorString(err));

            cudaDeviceSynchronize();
            distance >>= 1;
        }

        // Handle small distances (<=1024)
        exchange_V2<<<blocks_per_grid, threads_per_block>>>(array, size, group_size);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
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