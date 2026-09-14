#!/bin/bash
#SBATCH --job-name=capacity
#SBATCH --partition=20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-500%100
#SBATCH --output=logs/capacity_%A_%a.out

# Benchmark FastTree at num_extant=10000 across cassette count and missing rate.
# Grid: 9 cassette counts x 5 missing rates x 10 iterations = 450 array tasks.
# Each task maps $SLURM_ARRAY_TASK_ID -> one (cassettes, missing_rate, iteration).

set -euo pipefail

SIM_DIR="/lab/solexa_weissman/wcolgan/devmap-paper/simulation"
cd "$SIM_DIR"

# Activate the devmap conda environment.
source /lab/solexa_weissman/wcolgan/tools/miniconda3/etc/profile.d/conda.sh
conda activate devmap

# Keep numeric libraries from oversubscribing the allocated CPUs.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

python benchmark_capacity.py --outdir "$SIM_DIR/results_capacity"
