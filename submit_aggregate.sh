#!/bin/bash
#
#SBATCH --job-name=aggregate_partial_results
#SBATCH --output=res_aggregate.txt
#
#SBATCH --ntasks=1
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100

srun python aggregate.py