#!/bin/bash
#
#SBATCH --job-name=yeast_cleaned
#SBATCH --output=std_out/EQUAL_FREQUENCY/yeast_cleaned.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=10000
#
#SBATCH --array=0-39

srun python exp_equal_freq.py yeast_cleaned $SLURM_ARRAY_TASK_ID