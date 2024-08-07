#!/bin/bash -l
#SBATCH -p ecsstaff
#SBATCH --account=ecsstaff
#SBATCH --partition=a100
#SBATCH -c 48
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shj1g20@soton.ac.uk
#SBATCH --time=10:00:00

conda activate challenge

python feasibility_scripts/ppo_training.py
