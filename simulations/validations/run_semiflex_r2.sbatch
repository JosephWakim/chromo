#!/bin/bash

#SBATCH --job-name=semiflex
#SBATCH --time=11:59:59
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

source ~/.bashrc
conda activate chromo
cd /scratch/groups/ajspakow/WS_crosstalk_2024/chromo/simulations/validations
print("Starting simulation....")
python validate_end_to_end_distances.py
echo 'Job complete!'
