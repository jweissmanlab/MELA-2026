#!/bin/bash
#SBATCH --job-name=embed_eval
#SBATCH --output=models/logs/%A_%a.out
#SBATCH --error=models/logs/%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --array=0-8
#SBATCH --mem=256G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=ou_bcs_low

mkdir -p models/logs

# 8 scvi configs + 1 pca baseline = 9 jobs (indices 0-8)
# columns: mode  n_latent  n_layers
CONFIGS=(
    "0.0  20  1"
    "0.0  20  2"
    "0.0  50  1"
    "0.0  50  2"
    "1.0  20  1"
    "1.0  20  2"
    "1.0  50  1"
    "1.0  50  2"
    "pca  50  1"
)

CFG=(${CONFIGS[$SLURM_ARRAY_TASK_ID]})
MODE=${CFG[0]}
N_LATENT=${CFG[1]}
N_LAYERS=${CFG[2]}

echo "task $SLURM_ARRAY_TASK_ID: mode=$MODE n_latent=$N_LATENT n_layers=$N_LAYERS"

cd /home/gokulg/orcd/scratch/lt
python embed_and_eval.py $MODE $N_LATENT $N_LAYERS