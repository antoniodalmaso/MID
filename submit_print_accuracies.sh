#!/bin/bash
#
#SBATCH --job-name=print_accuracies_for_tables
#SBATCH --output=std_out/accuracies_for_tables.txt
#
#SBATCH --ntasks=1
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100

srun python print_accuracies_for_tables.py