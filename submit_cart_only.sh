#!/bin/bash
#
#SBATCH --job-name=cart_only
#SBATCH --output=std_out/cart_only.txt
#
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=10000

srun python exp_cart_only.py