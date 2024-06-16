#!/bin/bash
#
#SBATCH --job-name=aggregate_partial_results
#SBATCH --output=std_out/res_aggregate.txt
#
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=1000

srun python aggregate.py