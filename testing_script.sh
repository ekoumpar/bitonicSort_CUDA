#! /usr/bin/env bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00

module load gcc/12.2.0
module load cuda/12.2.1-bxtxsod

make V0
make run q=20
make clean
