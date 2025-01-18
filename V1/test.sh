#!/bin/bash
#SBATCH --partition=gpu


module load  gcc/13.2.0-iqpfkya cuda/12.4.0-zk32gam

nvcc main.cu bitonicSort.cu -o cuda_bitonic

./cuda_bitonic
