#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int compare(const void *a, const void *b)
{
    return (*(int *)a - *(int *)b);
}

int main(int argc, char *argv[])
{
    int q = atoi(argv[1]);
    size_t size = 1 << q; // 2^q elements

    int *array = (int *)malloc(size * sizeof(int));

    srand(time(0));
    for (int i = 0; i < size; i++)
    {
        array[i] = rand() % 1000; // random numbers from (0-1000)
    }

    clock_t start = clock();

    qsort(array, size, sizeof(int), compare);

    clock_t end = clock();

    double execution_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Execution time: %f seconds \n", execution_time);

    free(array);
    return 0;
}
