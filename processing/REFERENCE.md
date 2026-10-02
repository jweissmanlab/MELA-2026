# Building the custom Cell Ranger references

The processing scripts use two custom Cell Ranger references and a SNP panel. All of them are read from `REFERENCE_DIR`, which defaults to `../reference` (the repository `reference/` folder) when the scripts are run from `processing/`:

| File | Used by | Description |
|------|---------|-------------|
| `cellranger/mm39_PETS/` | `1_cellranger_gex.slurm` | GRCm39 genome with GENCODE vM37 annotations, plus PEtracer sequences |
| `cellranger/PETS/` | `2_cellranger_ts.slurm` | PEtracer sequences only, for target site libraries |
| `B6_SNP_panel.vcf` | `1_cellranger_gex.slurm` (VarTrix) | SNPs distinguishing host and donor cells (GRCm39 coordinates) |

`reference/PETS.fa` and `reference/PETS.gtf` contain the PEtracer sequences: the PEmax-T2A-GFP editor (`PE2maxGFP`) and the 2,171 possible lineage tracing cassettes (`intID1`–`intID2171`), each annotated as a single-exon gene. `reference/B6_SNP_panel.vcf` is provided in the repository.

## 1. Download GRCm39 and GENCODE vM37

```bash
cd reference
mkdir -p cellranger && cd cellranger
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M37/GRCm39.genome.fa.gz
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M37/gencode.vM37.annotation.gtf.gz
gunzip GRCm39.genome.fa.gz gencode.vM37.annotation.gtf.gz
```

## 2. Build the gene expression reference (`mm39_PETS`)

Append the PEtracer TS sequences to the mouse genome and annotations, then run `cellranger mkref`:

```bash
cat GRCm39.genome.fa ../PETS.fa > mm39_PETS.fa
cat gencode.vM37.annotation.gtf ../PETS.gtf > mm39_PETS.gtf
cellranger mkref --genome=mm39_PETS --fasta=mm39_PETS.fa --genes=mm39_PETS.gtf
rm mm39_PETS.fa mm39_PETS.gtf
```

## 3. Build the target site reference (`PETS`)

```bash
cellranger mkref --genome=PETS --fasta=../PETS.fa --genes=../PETS.gtf
```
