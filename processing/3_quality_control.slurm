#!/usr/bin/env bash
#SBATCH --job-name=qc 
#SBATCH --nodes=1  
#SBATCH --ntasks=1  
#SBATCH --cpus-per-task=8  
#SBATCH --mem=256gb 
#SBATCH --partition=weissman    
#SBATCH --account=weissman   
#SBATCH --output=logs/qc-%j.out 
#SBATCH --array=1-22%4 

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
min_umi="$(printf '%s' "$line" | awk -F',' '{gsub(/^[ \t]+|[ \t]+$/, "", $5); print $5}')"
min_reads="$(printf '%s' "$line" | awk -F',' '{gsub(/^[ \t]+|[ \t]+$/, "", $6); print $6}')"

echo "Running: n=$n p=$p c=$c min_umi=$min_umi min_reads=$min_reads"

# Disable adaptive thresholding for preliminary samples
extra_args=()
if (( SLURM_ARRAY_TASK_ID <= 6)); then
  extra_args+=(--min_total_counts 100 --disable_adaptive_threshold)
fi

python quality_control.py \
  -i "$INPUT_DIR" \
  -o "$OUTPUT_DIR" \
  -n "$n" \
  -p "$p" \
  -c "$c" \
  --min_umi "$min_umi" \
  --min_reads "$min_reads" \
  "${extra_args[@]}"

