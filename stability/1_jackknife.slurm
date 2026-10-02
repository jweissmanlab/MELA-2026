#!/usr/bin/env bash
#SBATCH --job-name=wcolgan_jackknife
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20gb
#SBATCH --partition=20
#SBATCH --account=weissman
#SBATCH --output=logs/jackknife-%A_%a.out
#SBATCH --time=08:00:00
#SBATCH --array=0-98

set -euo pipefail

# Leave-one-integration-out jackknife of the tree reconstruction. Each E9.5
# replicate has 33 integrations (99 characters, 3 per integration), so there
# are exactly 33 iterations per embryo (one dropped integration each).
# 3 embryos x 33 integrations = 99 array tasks. The array index selects the
# embryo (index / 33) and the integration to drop (index % 33).

EMBRYOS=("E9.5-R1" "E9.5-R2" "E9.5-R3")
N_INT=33
OUTPUT="./results"
TIME=9.5

NAME="${EMBRYOS[$((SLURM_ARRAY_TASK_ID / N_INT))]}"
INT_INDEX=$((SLURM_ARRAY_TASK_ID % N_INT))
PREFIX="${NAME,,}"; PREFIX="${PREFIX//-/_}"
INPUT="./data/${PREFIX}_stump.h5td"

echo "Jackknife: ${NAME} dropping integration index ${INT_INDEX} (${INPUT})"

python jackknife.py \
  -i "$INPUT" \
  -o "$OUTPUT" \
  -n "$NAME" \
  -t "$TIME" \
  -g "$INT_INDEX" \
  -p "$SLURM_CPUS_PER_TASK"
