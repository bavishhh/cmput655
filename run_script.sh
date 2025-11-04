#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --account=def-machado
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6:0:0
#SBATCH --mem=8GB
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --job-name=cmput655_test
#SBATCH --output=cmput655_test.out
#SBATCH --error=cmput655_test.err

module load python/3.11.5
module load cuda/12.2

python /home/bavish/projects/def-machado/bavish/cmput655/train.py

