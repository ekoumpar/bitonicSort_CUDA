#ifndef BITONIC_H
#define BITONIC_H

#include <stddef.h>

__global__ void exchangeKernel(int *array, int size, int group_size, int distance);

void bitonicSort(int **array, int size);
void swap(int *a, int *b);
void print(int *array, int size);
void evaluateResult(int *array, int size);


#endif
