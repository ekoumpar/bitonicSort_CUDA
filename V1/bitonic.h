#ifndef BITONIC_H
#define BITONIC_H

#include <stddef.h>

__device__ void swap(int *array, int idx, int partner);
__global__ void initialExchange(int *array, int size);
__global__ void exchange_V0(int *array, int size, int group_size, int distance);
__global__ void exchange_V1(int *array, int size, int group_size);

void bitonicSort(int *array, int size);
void print(int *array, int size);
void evaluateResult(int *array, int size);

#endif
