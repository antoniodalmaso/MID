#!/bin/bash
#
#SBATCH --job-name=plot_results
#SBATCH --output=std_out/res_plot.txt
#
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=1000

srun python plot.py