#!/bin/bash
#SBATCH --time=03:30:00
#SBATCH --array=0-539
#SBATCH --account=aip-lelis
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --ntasks=1
#SBATCH --job-name GreedyAC-hypers-sweep-1of6
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gunuputi@ualberta.ca
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j-err.out

module load cuda/12.9 scipy-stack python/3.13.2
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install click
pip install gymnasium
pip install gym
pip install wandb
pip install torch
pip install minatar
pip install bootstrapped==0.0.2
pip install tqdm
python main.py --index $SLURM_ARRAY_TASK_ID --env-json "/home/saigp/scratch/cmput655/GreedyAC-master/config/environment/MountainCarContinuous-v0.json" --agent-json "/home/saigp/scratch/cmput655/GreedyAC-master/config/agent/GreedyAC.json"