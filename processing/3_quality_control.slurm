#!/usr/bin/env bash
#SBATCH --job-name=qc          # friendly name for job.
#SBATCH --nodes=1                      # ensure cpus are on one node
#SBATCH --ntasks=1                     # run a single task
#SBATCH --cpus-per-task=8             # number of cpus/threads requested.
#SBATCH --mem=256gb                    # memory requested.
#SBATCH --partition=weissman               # partition (queue) to use
#SBATCH --account=weissman                 # account to use
#SBATCH --output=logs/qc-%j.out     # name of output file.  %j is jobid
#SBATCH --array=10-12                    # array of jobs to submit

set -euo pipefail

CSV="samples.csv"   # columns: n,p,c,t (with a header row)
INPUT_DIR="./data"
OUTPUT_DIR="./embryos"

# Read the Nth data row (skip header). Task 1 -> first data row.
line="$(awk -F',' -v row="$SLURM_ARRAY_TASK_ID" 'NR==row+1 {print; exit}' "$CSV")"

if [[ -z "${line}" ]]; then
  echo "No CSV row found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

# Parse n,p,c,t (trim whitespace; assumes fields do not contain commas)
n="$(printf '%s' "$line" | awk -F',' '{gsub(/^[ \t]+|[ \t]+$/, "", $1); print $1}')"
p="$(printf '%s' "$line" | awk -F',' '{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}')"
c="$(printf '%s' "$line" | awk -F',' '{gsub(/^[ \t]+|[ \t]+$/, "", $3); print $3}')"

echo "Running: n=$n p=$p c=$c"

python quality_control.py \
  -i "$INPUT_DIR" \
  -o "$OUTPUT_DIR" \
  -n "$n" \
  -p "$p" \
  -c "$c" \

