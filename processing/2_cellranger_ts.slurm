#!/bin/bash
#SBATCH --job-name=cellranger_ts  
#SBATCH --nodes=1   
#SBATCH --ntasks=1  
#SBATCH --cpus-per-task=16  
#SBATCH --mem=200gb  
#SBATCH --partition=20  
#SBATCH --account=weissman  
#SBATCH --output=cellranger_ts-%j.out
#SBATCH --array=1-117%10

set -euo pipefail

## Configure
CSV="captures.csv"
transcriptome="/lab/solexa_weissman/wcolgan/reference/cellranger/PETS"
transcriptome=$(realpath "$transcriptome")
fastq_dir=$(realpath "./fastq")

# Read the Nth data row (skip header). Task 1 -> first data row.
line="$(awk -F',' -v row="$SLURM_ARRAY_TASK_ID" 'NR==row+1 {print; exit}' "$CSV")"
if [[ -z "${line}" ]]; then
  echo "No CSV row found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
  exit 1
fi
sample="$(printf '%s' "$line" | awk -F',' '{gsub(/^[ \t]+|[ \t]+$/, "", $1); print $1}')"

# Template for the libraries.csv content
libraries_template="fastqs,sample,library_type
{fastq_dir},{sample}_TS,Gene Expression"

#Make output directories
mkdir -p ./bam
mkdir -p ./data

# Run cellranger count
echo "Running cellranger count for sample: ${sample}"
libraries_csv="${sample}_libraries.csv"
echo "$libraries_template" | sed "s/{sample}/$sample/g" | sed "s|{fastq_dir}|$fastq_dir|g" > "$libraries_csv"
cellranger count \
  --id="$sample" \
  --libraries="$libraries_csv" \
  --transcriptome="$transcriptome" \
  --create-bam=true \
  --localmem=200 \
  --nosecondary

# Clean up
rm "$libraries_csv"
output_dir="./$sample/outs"
data_dir="./data/${sample}"
mkdir -p $data_dir
# Bam
bam_file="${output_dir}/possorted_genome_bam.bam"
mv "$bam_file" "./bam/${sample}_ts.bam"
bai_file="${output_dir}/possorted_genome_bam.bam.bai"
mv "$bai_file" "./bam/${sample}_ts.bai"
# Counts
filtered_h5="${output_dir}/filtered_feature_bc_matrix.h5"
mv "$filtered_h5" "${data_dir}/${sample}_ts_filtered_counts.h5"
# Summary
summary="${output_dir}/web_summary.html"
mv "$summary" "${data_dir}/${sample}_ts_summary.html"
# Remove output directory
rm -r "./${sample}"

# Call TS alleles
echo "Calling TS alleles for sample: ${sample}"
tracertools alleles-from-bam \
  --bam "./bam/${sample}_ts.bam" \
  --out "./data/"${sample}"/${sample}"_allele_counts.csv \
  --barcode_start 270 \
  --barcode_end 300 \
  --site_positions  "{'RNF2':332,'HEK3':380,'EMX1':448}" \
  --min_reads 2

rm "./bam/${sample}.bam"
