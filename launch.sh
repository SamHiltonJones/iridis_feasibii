#!/bin/bash -l
#SBATCH -p ecsstaff
#SBATCH --account=ecsstaff
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shj1g20@soton.ac.uk
#SBATCH --time=10:00:00

conda activate feasibility

python feasibility_scripts/simple_model_linux.py
