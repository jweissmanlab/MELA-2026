# The Quantitative Lineage Architecture of Mouse Embryogenesis

This repository contains code accompanying the paper **"The Quantitative Lineage Architecture of Mouse Embryogenesis"**.

## Repository Structure

```
├── devmap/                  # Core Python package
├── processing/              # Raw data processing and lineage tree reconstruction
├── annotation/              # Cell type annotation and clustering
├── validation/              # Validation of the PEtracer system
├── embedding/               # scVI embedding and evaluation
├── fate/                    # Cell fate bias and germ layer analysis
├── gene_programs/           # Gene program identification
├── axial_patterning/        # Anterior-posterior and neural patterning
├── vignettes/               # Tissue deep-dives (heart, neural crest, notochord)
├── commitment/              # Node-level commitment analysis (MMD)
├── distributional_metrics/  # Clonal and embryo-level distributional distances
└── staging/                 # Cross-dataset integration (optimal transport)
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

## Annotation

Notebooks and scripts for annotating cell types. Steps should be run in order from the `annotation/` directory.

**Software:** [CellxGene](https://github.com/chanzuckerberg/cellxgene) v1.2.0, Claude CLI v2.1.138

| Step | Script | Description |
|------|--------|-------------|
| 1 | `1_clusters_and_markers.ipynb` | Combine embryos, compute HVGs, cluster cells, identify marker genes |
| 2 | `2_initial_claude_annotation.py` | Annotate initial clusters with Claude based on markers |
| 3 | `3_refine_with_cellxgene.slurm` | Launch CellxGene instances per cluster for manual annotation refinement |
| 4 | `4_cleanup_and_markers.ipynb` | Drop doublets and identify cell subtype marker genes |
| 5 | `5_final_claude_annotation.py` | Annotate final cell subtypes with Claude based on markers |

Steps 2 and 5 call the Claude CLI. Step 3 requires interactive use of CellxGene; manually curated labels are saved to `data/cell_types.csv` and read by step 4.

## Validation

Notebooks validating the PEtracer system and tree reconstruction accuracy. Uses preliminary bulk and scRNA-seq data as well as full scRNA-seq dataset. Steps should be run in order from the `validation/` directory.

| Step | Script | Description |
|------|--------|-------------|
| 1 | `1_ideogram.html` | Ideogram of PEtracer target site genomic locations |
| 2 | `2_bulk_kinetics.ipynb` | Bulk sequencing to determine editing kinetics in vitro and in vivo |
| 3 | `3_preliminary_tracing.ipynb` | Analysis of preliminary scRNA-seq data |
| 4 | `4_tracing_performance.ipynb` | Tracing performance evaluation in the full dataset |

## Gene Programs

Notebooks and scripts for identifying and annotating heritable gene programs. Steps should be run in order from the `gene_programs/` directory.

**Software:** Claude CLI v2.1.138

| Step | Script | Description |
|------|--------|-------------|
| 1 | `1_define_programs.ipynb` | Identify heritable gene expression programs using Hotspot |
| 2 | `2_annotate_programs.py` | Annotate programs with Claude |
| 3 | `3_analyze_programs.ipynb` | Visualize program activation and rate of change|

