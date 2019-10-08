#!/bin/bash
#SBATCH --job-name=cython
#SBATCH --output=/home/livingstonb/GitHub/DT_Consumption_Adj_Costs/output/run%a.out
#SBATCH --error=/home/livingstonb/GitHub/DT_Consumption_Adj_Costs/output/run%a.err
#SBATCH --partition=broadwl
#SBATCH --array=0-18
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=10000

python -u master.py $SLURM_ARRAY_TASK_ID