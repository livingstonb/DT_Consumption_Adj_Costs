#!/bin/bash
#SBATCH --job-name=cython
#SBATCH --output=/home/livingstonb/GitHub/DT_Consumption_Adj_Costs/output/run%a.out
#SBATCH --error=/home/livingstonb/GitHub/DT_Consumption_Adj_Costs/output/run%a.err
#SBATCH --partition=broadwl
#SBATCH --array=2
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=22000

python -u /home/livingstonb/GitHub/DT_Consumption_Adj_Costs/master.py $SLURM_ARRAY_TASK_ID