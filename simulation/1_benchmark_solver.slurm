#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --partition=20
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

# Run from the simulation/ directory (sbatch 1_benchmark_solver.slurm).
SIM_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$SIM_DIR"

# Activate the devmap conda environment.
eval "$(conda shell.bash hook)"
conda activate devmap

# Keep numeric libraries from oversubscribing the allocated CPUs.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

python benchmark_solver.py --outdir "$SIM_DIR/results_solver"
