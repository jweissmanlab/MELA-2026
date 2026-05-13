#!/bin/bash
#SBATCH --job-name=linkage
#SBATCH --output=logs/linkage-%j.out
#SBATCH --array=0-15
#SBATCH --cpus-per-task=32
#SBATCH --mem=100gb 
#SBATCH --partition=20
#SBATCH --account=weissman

EMBRYOS=(
 "E7.5-R1"
 "E7.5-R2"
 "E7.5-R3"
 "E8.0-R1"
 "E8.0-R2"
 "E8.0-R3"
 "E8.5-R1"
 "E8.5-R2"
 "E8.5-R3"
 "E9.0-R1"
 "E9.0-R2"
 "E9.0-R3"
 "E9.5-R1"
 "E9.5-R2"
 "E9.5-R3"
 "E10.0-R1")

EMBRYO="${EMBRYOS[$SLURM_ARRAY_TASK_ID]}"

echo "Calculating celltype linkage for ${EMBRYO}"
python embryo_linkage.py "${EMBRYO}" 

echo "Calculating subtype linkage for ${EMBRYO}"
python embryo_linkage.py "${EMBRYO}" --groupby cell_subtype 
