#ifndef BITONIC_H
#define BITONIC_H

#include <stddef.h>

__device__ void swap(int *array, int idx, int partner);
__device__ void load_global_to_local(int *array, int *local_array, int size, int local_tid, int offset);
__device__ void load_local_to_global(int *array, int *local_array, int size, int local_tid, int offset);
__global__ void initialExchange(int *array, int size);
__global__ void exchange_V0(int *array, int size, int group_size, int distance);
__global__ void exchange_V2(int *array, int size, int group_size);

void bitonicSort(int *array, int size);
void print(int *array, int size);
void evaluateResult(int *array, int size);

#endif
