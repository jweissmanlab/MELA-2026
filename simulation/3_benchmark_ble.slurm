#!/bin/bash
#SBATCH --job-name=ble
#SBATCH --partition=24
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-499%100
#SBATCH --output=logs/ble_%A_%a.out

# Benchmark ConvexML branch-length estimation at num_extant=10000 across
# division_sigma and cassette count.
# Grid: 10 cassette counts x 5 sigmas x 10 iterations = 500 array tasks.
# Each task maps $SLURM_ARRAY_TASK_ID -> one (cassettes, sigma, iteration).

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

python benchmark_ble.py --outdir "$SIM_DIR/results_ble"
