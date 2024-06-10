#!/bin/bash
#
#SBATCH --job-name=plot_results
#SBATCH --output=res_plot.txt
#
#SBATCH --ntasks=1
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100

srun python plot.py