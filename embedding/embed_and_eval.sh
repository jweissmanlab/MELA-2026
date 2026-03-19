#!/bin/bash
#SBATCH --job-name=embed_eval
#SBATCH --output=models/logs/%A_%a.out
#SBATCH --error=models/logs/%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --array=0-3%2
#SBATCH --mem=256G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=bates

mkdir -p models/logs

# columns: mode  n_latent  cov
CONFIGS=(
    "0.0  50  none"
    "0.0  50  type"
    "1.0  50  none"
    "1.0  50  type"
)

CFG=(${CONFIGS[$SLURM_ARRAY_TASK_ID]})
MODE=${CFG[0]}
N_LATENT=${CFG[1]}
COV=${CFG[2]}

echo "task $SLURM_ARRAY_TASK_ID: mode=$MODE n_latent=$N_LATENT cov=$COV"

cd /mnt/home/gokulg/devmap-paper/
python embedding/embed_and_eval.py $MODE $N_LATENT $COV