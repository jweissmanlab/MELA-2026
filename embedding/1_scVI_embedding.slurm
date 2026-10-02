#!/bin/bash
#SBATCH --job-name=train_embedding
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --partition=nvidia-A100-20

mkdir -p logs
python scVI_embedding.py
