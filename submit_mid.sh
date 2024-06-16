#!/bin/bash
#
#SBATCH --job-name=spambase_6
#SBATCH --output=std_out/MID/spambase_6.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=10000
#
#SBATCH --array=0-9

srun python exp_mid.py spambase 6 5 $SLURM_ARRAY_TASK_ID