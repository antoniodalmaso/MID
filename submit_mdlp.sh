#!/bin/bash
#
#SBATCH --job-name=pendigits
#SBATCH --output=std_out/MDLP/pendigits.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=10000
#
#SBATCH --array=0-39

srun python exp_mdlp.py pendigits $SLURM_ARRAY_TASK_ID