#!/bin/bash
#
#SBATCH --job-name=penguins_cleaned
#SBATCH --output=std_out/MDLP/penguins_cleaned.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=10000
#
#SBATCH --array=0-39

srun python exp_mdlp.py penguins_cleaned $SLURM_ARRAY_TASK_ID