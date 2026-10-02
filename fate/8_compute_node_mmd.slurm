#!/bin/bash
#SBATCH --job-name=mmd_array
#SBATCH --output=logs/mmd_%A_%a.out
#SBATCH --error=logs/mmd_%A_%a.err
#SBATCH --array=0-55%8
#SBATCH --time=02:00:00 
#SBATCH --partition=nvidia-A100-20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL

set -euo pipefail
mkdir -p logs results/node_mmd

echo "Array job $SLURM_ARRAY_JOB_ID, task $SLURM_ARRAY_TASK_ID on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python compute_node_mmd.py \
    --adata ../data/topology.h5td \
    --tree-dir ../processing/embryos \
    --outdir results/node_mmd \
    --tree-index "$SLURM_ARRAY_TASK_ID" \
    --n-subsample 200 \
    --n-resamples 5 \
    --n-perms 10 \
    --min-descendants 5

echo "Task $SLURM_ARRAY_TASK_ID finished at $(date)"