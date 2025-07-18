#!/bin/bash
#SBATCH --job-name=phi4-finetune
#SBATCH --output=logs/phi4-%j.out
#SBATCH --error=logs/phi4-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu

# === Load Conda ===
source ~/.bashrc
conda activate phi4env

# === Optional: navigate to your project folder ===
cd /users/sgvanil/your_project_folder  # change to actual folder if needed

# === Run your training script ===
python train_phi4.py