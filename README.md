# The Quantitative Lineage Architecture of Mouse Embryogenesis

This repository contains code accompanying the paper **"The Quantitative Lineage Architecture of Mouse Embryogenesis"**. The analyses reconstruct a lineage-traced atlas of mouse embryogenesis (E7.5–E10.0).

## Repository Structure

```
├── devmap/                  # Core Python package
├── processing/              # Raw data QC and lineage tree reconstruction
├── annotation/              # Cell type annotation and clustering
├── fate/                    # Cell fate bias and germ layer analysis
├── gene_programs/           # Gene program identification
├── axial_patterning/        # Anterior-posterior and neural patterning
├── vignettes/               # Tissue deep-dives (heart, neural crest, notochord)
├── commitment/              # Node-level commitment analysis (MMD)
├── distributional_metrics/  # Clonal and embryo-level distributional distances
├── embedding/               # scVI embedding and evaluation
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

## Analysis Pipeline

The processing pipeline runs on a Slurm cluster. Steps should be run in order:

| Step | Script | Description |
|------|--------|-------------|
| 3 | `processing/3_quality_control.slurm` | Per-embryo quality control |
| 4 | `processing/4_reconstruct.slurm` | Lineage tree reconstruction |
| 5 | `annotation/annotate_clusters.slurm` | Cluster annotation |

Downstream analyses (fate, gene programs, vignettes, etc.) are in self-contained Jupyter notebooks within their respective directories.

## Data Availability

Processed data files (`.h5td`, `.h5ad`) are available from [GEO/Zenodo — link TBD]. Place them in the `data/` directory before running notebooks.

## Citation

> Koblan, Colgan et al. "The Quantitative Lineage Architecture of Mouse Embryogenesis." *bioRxiv* (2026).
