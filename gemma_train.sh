#!/bin/bash
#SBATCH --job-name=gemma13b-qlora
#SBATCH --output=logs/gemma13b-%j.out
#SBATCH --error=logs/gemma13b-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-a100  # ✅ Make sure this is the A100 (preferably 80GB)

source ~/.bashrc
conda activate gemmaenv

cd /users/sgvanil
mkdir -p logs

export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export CUDA_VISIBLE_DEVICES=0

python -u gemma_train.py
