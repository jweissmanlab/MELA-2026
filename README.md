# The Quantitative Lineage Architecture of Mouse Embryogenesis

This repository contains code accompanying the paper **"The Quantitative Lineage Architecture of Mouse Embryogenesis"**.

## Repository Structure

```
├── devmap/                  # Core Python package
├── processing/              # Raw data processing and lineage tree reconstruction
├── annotation/              # Cell type annotation and clustering
├── embedding/               # scVI embedding and evaluation
├── fate/                    # Cell fate bias and germ layer analysis
├── gene_programs/           # Gene program identification
├── axial_patterning/        # Anterior-posterior and neural patterning
├── vignettes/               # Tissue deep-dives (heart, neural crest, notochord)
├── commitment/              # Node-level commitment analysis (MMD)
├── distributional_metrics/  # Clonal and embryo-level distributional distances
├── integration/             # Cross-dataset integration (optimal transport)
└── validation/              # Validation analyses
```

## Setup

Requires [mamba](https://mamba.readthedocs.io/) (or conda).

```bash
mamba env create --file environment.yml
mamba activate devmap-paper
ipython kernel install --user --name devmap-paper
```

To update an existing environment:

```bash
mamba env update --file environment.yml
```

## Data Availability

Processed data files (`.h5td`) are available from Zenodo [link TBD]. Place them in the `data/` directory before running analysis. To regenerate processed data, fastq files can be downloaded from SRA [link TBD] and placed in `processing/fastq/`.

## Processing

Scripts for running Cellranger, quality control, and lineage tree reconstruction. Steps 1–4 run as Slurm job arrays; step 5 is a Jupyter notebook. All steps must be run in order from the `processing/` directory.

**Software:** [Cellranger](https://github.com/10XGenomics/cellranger) v9.0.1, [VarTrix](https://github.com/10xgenomics/vartrix) v1.1.22, [tracertools](https://github.com/colganwi/tracertools) v0.2.0

| Step | Script | Description |
|------|--------|-------------|
| 1 | `1_cellranger_gex.slurm` | Align gene expression reads with Cellranger; run VarTrix for SNP genotyping |
| 2 | `2_cellranger_ts.slurm` | Align target site reads with Cellranger; call PEtracer alleles |
| 3 | `3_quality_control.slurm` | Per-embryo QC including doublet detection and allele processing |
| 4 | `4_reconstruct.slurm` | Lineage tree reconstruction |
| 5 | `5_embryo_statistics.ipynb` | Calculate per-embryo topology and recording statistics |

```bash
cd processing/
sbatch 1_cellranger_gex.slurm
sbatch 2_cellranger_ts.slurm
sbatch 3_quality_control.slurm
sbatch 4_reconstruct.slurm
```

## Annotation

Notebooks and scripts for clustering and annotating cell types.

## Embedding

Scripts and notebooks for training the scVI embedding and evaluating it.

## Fate

Notebooks for cell fate bias and germ layer analysis.

## Gene Programs

Notebooks for identifying and annotating gene programs.

## Axial Patterning

Notebooks for anterior-posterior scoring and neural patterning analysis.

## Vignettes

Tissue-level deep-dives into heart, neural crest, and notochord development.

## Commitment

Node-level lineage commitment analysis using maximum mean discrepancy (MMD).

## Distributional Metrics

Pairwise clonal and embryo-level distributional distance calculations.

## Integration

Cross-dataset integration with external reference datasets using optimal transport.

## Validation

Validation analyses including label transfer from reference atlases.

---

## Citation

> Koblan, Colgan et al. "The Quantitative Lineage Architecture of Mouse Embryogenesis." *bioRxiv* (2026).
