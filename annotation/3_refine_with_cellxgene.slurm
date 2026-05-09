#!/bin/bash
#SBATCH --job-name=cellxgene
#SBATCH --output=logs/cellxgene_%A_%a.out
#SBATCH --array=0-20
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=weissman
#SBATCH --account=weissman
#SBATCH --nodelist=c4b15
#SBATCH --time=7-00:00:00

# Load your environment (adjust as needed)
source activate cellxgene

# Compute cluster and port
CLUSTER_ID=${SLURM_ARRAY_TASK_ID}
PORT=$((8020 + CLUSTER_ID))

# Build filename
FILE="/lab/solexa_weissman/fame/data/E95_annotated_v3_cluster_${CLUSTER_ID}.h5ad"

echo "Launching CellxGene for cluster ${CLUSTER_ID} on port ${PORT}"
echo "File: ${FILE}"

cellxgene launch "$FILE" --host 0.0.0.0 --port "${PORT}"