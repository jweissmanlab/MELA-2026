#!/bin/bash
#SBATCH --job-name=mmd_array
#SBATCH --output=logs/mmd_%A_%a.out
#SBATCH --error=logs/mmd_%A_%a.err
#SBATCH --array=0-55%8
#SBATCH --time=02:00:00 
#SBATCH --partition=ou_bcs_low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL

set -euo pipefail
mkdir -p logs commitment/per_tree

echo "Array job $SLURM_ARRAY_JOB_ID, task $SLURM_ARRAY_TASK_ID on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python commitment/compute_node_mmd.py \
    --adata ../lt/data/kl0.0_d50_l2_covnocov_adata_with_embeddings.h5ad \
    --tree-dir ../lt/data/embryos \
    --outdir commitment/per_tree \
    --tree-index "$SLURM_ARRAY_TASK_ID" \
    --n-subsample 200 \
    --n-resamples 5 \
    --n-perms 10 \
    --min-descendants 5

echo "Task $SLURM_ARRAY_TASK_ID finished at $(date)"