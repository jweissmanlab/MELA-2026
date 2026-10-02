#!/usr/bin/env bash
#SBATCH --job-name=wcolgan_permutation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --partition=20
#SBATCH --account=weissman
#SBATCH --output=logs/permutation-%A_%a.out
#SBATCH --time=03:00:00
#SBATCH --array=0-9

set -euo pipefail

# Simulated leaf-misplacement stability test. Each array task simulates one
# ground-truth tree topology (seed = array index) and applies every
# misplacement rate to that same tree, so the topology and fates are identical
# across rates. 10 topologies x rates {0,2,4,6,8,10,12,14,16,18,20}.

OUTPUT="./results"
RATES="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"

echo "Simulation topology ${SLURM_ARRAY_TASK_ID}, rates: ${RATES}"

python permutation.py \
  -o "$OUTPUT" \
  -s "$SLURM_ARRAY_TASK_ID" \
  -r $RATES
