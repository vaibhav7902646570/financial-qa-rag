#!/bin/bash
#SBATCH --job-name=gemma-infer
#SBATCH --output=gemmatest_output.txt
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=gpu-a100-lowbig

# Activate your conda environment
source ~/.bashrc
conda activate phi4env

# Run the script
python -u gemmatest.py
