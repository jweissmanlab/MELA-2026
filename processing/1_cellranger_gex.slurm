#!/bin/bash
#SBATCH --job-name=cellranger_gex  
#SBATCH --nodes=1    
#SBATCH --ntasks=1   
#SBATCH --cpus-per-task=16    
#SBATCH --mem=150gb      
#SBATCH --partition=20      
#SBATCH --account=weissman    
#SBATCH --output=cellranger_gex-%j.out  
#SBATCH --array=1-117%10   

set -euo pipefail

## Configure
CSV="captures.csv"
transcriptome="/lab/solexa_weissman/wcolgan/reference/cellranger/mm39_PETS"
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
{fastq_dir},{sample}_GEX,Gene Expression"

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
  --localmem=150 \
  --nosecondary 

output_dir="./$sample/outs"
data_dir="./data/${sample}"
mkdir -p $data_dir

# Run vartrix
rm -f "${data_dir}/${sample}_alt_counts.mtx"
rm -f "${data_dir}/${sample}_ref_counts.mtx"
vartrix -b "${output_dir}/possorted_genome_bam.bam" \
    -c "${output_dir}/filtered_feature_bc_matrix/barcodes.tsv.gz" \
    -v "/lab/solexa_weissman/wcolgan/reference/B6_SNP_panel.vcf" \
    -f "${transcriptome}/fasta/genome.fa" \
    --out-matrix "${data_dir}/${sample}_alt_counts.mtx" \
    --ref-matrix "${data_dir}/${sample}_ref_counts.mtx" \
    --scoring-method coverage

# Clean up
rm -f "$libraries_csv"
# Bam
bam_file="${output_dir}/possorted_genome_bam.bam"
mv -f "$bam_file" "./bam/${sample}.bam"
bai_file="${output_dir}/possorted_genome_bam.bam.bai"
mv -f "$bai_file" "./bam/${sample}.bam.bai"
# Counts
filtered_h5="${output_dir}/filtered_feature_bc_matrix.h5"
mv -f "$filtered_h5" "${data_dir}/${sample}_filtered_counts.h5"
# Barcodes
barcodes="${output_dir}/filtered_feature_bc_matrix/barcodes.tsv.gz"
mv -f "$barcodes" "${data_dir}/${sample}_barcodes.tsv.gz"
# Summary
summary="${output_dir}/web_summary.html"
mv -f "$summary" "${data_dir}/${sample}_summary.html"
# Remove output directory
rm -r "./${sample}"
rm -f "./bam/${sample}.bam"
rm -f "./bam/${sample}.bam.bai"