#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --partition=24
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-600%100
#SBATCH --output=logs/benchmark_%A_%a.out

# Benchmark lineage tree reconstruction solvers.
# Grid: 5 sizes x 5 solvers x 10 iterations = 250 array tasks.
# Each task maps $SLURM_ARRAY_TASK_ID -> one (size, solver, iteration).

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

python benchmark.py --outdir "$SIM_DIR/results"
