# Mouse Embryonic Lineage Atlas

This repository contains the code accompanying Colgan, Koblan et al. 2026, **"Comprehensive Lineage Tracing Maps the Landscape of Cell Fate Decisions in Mouse Embryogenesis"**.

The lineage trees, gene expression, and fate restriction dynamics can be explored interactively at **[mela.wi.mit.edu](https://mela.wi.mit.edu)**.

## Overview

We used the PEtracer prime editing–based lineage recording system to continuously install heritable lineage marks across >100 editable sites in chimeric mouse embryos. Sixteen whole embryos were dissociated in their entirety and profiled by scRNA-seq at half-day intervals from E7.5 to E10.0, with triplicate sampling at all but the final timepoint. We recovered >1.7 million cells and reconstructed lineage trees for >1.4 million donor cells that resolve ~75% of cell divisions. Pairing these trees with deep transcriptional profiling, we quantify cell fate bias, the timing of fate restriction, progenitor pool sizes, and lineage relationships across 99 cell types and 142 subtypes. These features are strikingly reproducible between replicate embryos. Vignettes chart the lineage dynamics of the notochord, heart fields, neural crest, endothelium, neural ectoderm regionalization, and axial elongation. We also identify heritable gene programs and infer anterior-posterior and dorsal-ventral spatial positions to relate lineage to spatial patterning.

## Repository structure

```
├── devmap/              # Core Python package (data loading, fate, linkage, plotting utilities)
├── data/                # Data directory (see Data availability); merge_embryos.ipynb
├── processing/          # Raw data processing and lineage tree reconstruction
├── annotation/          # Cell type annotation and clustering
├── validation/          # Validation of the PEtracer system and tracing performance
├── embedding/           # scVI and UMAP embeddings
├── staging/             # Integration with reference atlases and transcriptional staging
├── simulation/          # In silico benchmarking of reconstruction and branch-length estimation
├── stability/           # Robustness to integration dropout and topological error
├── fate/                # Lineage trees, fate restriction, ancestral linkage, and fate bias
├── gene_programs/       # Heritable gene program identification
├── spatial_patterning/  # Spatial score inference and neural patterning
└── vignettes/           # Notochord, heart, neural crest, endothelial, and axial progenitor analyses
```

## Setup

Requires [mamba](https://mamba.readthedocs.io/) (or conda). The main analysis environment is `devmap`:

```bash
mamba env create --file environment.yml
# mamba env create --file environment.lock.yml  # exact versions
mamba activate devmap
ipython kernel install --user --name devmap
```

Steps that use [rapids-singlecell](https://github.com/scverse/rapids_singlecell) or [scvi-tools](https://scvi-tools.org), including the staging analyses, run in a separate `gpu` environment. This environment needs an NVIDIA GPU with CUDA 12:

```bash
mamba env create --file environment.gpu.yml
mamba activate gpu
ipython kernel install --user --name gpu
```

Steps that require the `gpu` environment are marked **[gpu]** below.

### Additional dependencies

These tools are not included in the conda environments and must be installed separately:

- [Cell Ranger](https://www.10xgenomics.com/support/software/cell-ranger) v9.0.1 (processing)
- [VarTrix](https://github.com/10xgenomics/vartrix) v1.1.22 (processing)
- [FastTree](http://www.microbesonline.org/fasttree/) v2.1.11 (tree reconstruction in processing, stability, and simulation)
- [cellxgene](https://github.com/chanzuckerberg/cellxgene) v1.3.0 (annotation step 3)
- [Claude Code CLI](https://docs.claude.com/en/docs/claude-code) v2.1.138 (annotation steps 2 and 5, fate step 3, gene programs step 2)
- [LAML-Pro](https://github.com/raphael-group/LAML-Pro) (simulation only, see below)

### Running on Slurm

The `.slurm` scripts were written for the Whitehead Institute cluster. Before submitting, edit the `#SBATCH` headers for your cluster:

- CPU jobs use `--partition=20`. GPU jobs (`embedding/1_scVI_embedding.slurm` and `fate/8_compute_node_mmd.slurm`) use `--partition=nvidia-A100-20` with `--gres=gpu:1`.
- Several scripts set `--account=weissman`. Change or remove this line for your cluster.

## Data availability

- **Raw data:** scRNA-seq and target site sequencing data are available from GEO under accession [GSE342607](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE342607).
- **Processed data:** per-embryo TreeData (`.h5td`) files and derived tables are available from Zenodo at [10.5281/zenodo.19892784](https://doi.org/10.5281/zenodo.19892784).
- **Code:** this repository is archived on Zenodo at [10.5281/zenodo.22833553](https://doi.org/10.5281/zenodo.22833553).

### Preparing the processed data

The processing steps can be skipped by downloading the processed data from [Zenodo](https://doi.org/10.5281/zenodo.19892784):

1. Place all of the per-embryo `.h5td` files (e.g. `E9.5-R1.h5td`) in the `data/embryos/` folder.
2. Run `data/merge_embryos.ipynb`. It generates the intermediates that every downstream analysis loads through `devmap.utils.load_data`:
   - `counts.h5td`, `log1p.h5td`, `log1p_hvg.h5td`: combined raw, normalized, and highly-variable-gene expression
   - `topology.h5td`: lineage trees and cell metadata without expression
   - `obs.csv`, `scvi.csv`, `umap.csv`: cell metadata and embeddings

The expression files are large (100+ GB), so run this notebook on a high-memory node.

## Processing

Scripts for running Cell Ranger, quality control, and lineage tree reconstruction. Raw fastq files can be downloaded from GEO ([GSE342607](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE342607)) and placed in `processing/fastq/`. All steps run as Slurm job arrays and must be run in order from the `processing/` directory. The custom Cell Ranger references (GRCm39 plus PEtracer sequences) and the SNP panel VCF are read from `REFERENCE_DIR` (default `../reference`). See [processing/REFERENCE.md](processing/REFERENCE.md) for instructions on building the references.

**This section can be skipped by downloading the processed data from Zenodo (see [Preparing the processed data](#preparing-the-processed-data)).**

**Software:** [Cell Ranger](https://github.com/10XGenomics/cellranger) v9.0.1, [VarTrix](https://github.com/10xgenomics/vartrix) v1.1.22, [tracertools](https://github.com/colganwi/tracertools) v0.3.0, [FastTree](http://www.microbesonline.org/fasttree/) v2.1.11

| Step | Script | Description |
|------|--------|-------------|
| 1 | `1_cellranger_gex.slurm` | Align gene expression reads with Cell Ranger and genotype host/donor SNPs with VarTrix |
| 2 | `2_cellranger_ts.slurm` | Align target site reads with Cell Ranger and call PEtracer alleles with tracertools |
| 3 | `3_quality_control.slurm` | Per-embryo QC: low-quality cell filtering, allele filtering, donor/host calling, and doublet detection (runs `quality_control.py`) |
| 4 | `4_reconstruct.slurm` | Lineage tree reconstruction and branch-length estimation (runs `reconstruct.py`) |
| – | `samples.csv`, `captures.csv` | Per-embryo and per-capture sample sheets used by the Slurm arrays |

## Annotation

Notebooks and scripts for annotating cell types. Steps should be run in order from the `annotation/` directory.

**Software:** [cellxgene](https://github.com/chanzuckerberg/cellxgene) v1.3.0, Claude Code CLI v2.1.138

| Step | Script | Description |
|------|--------|-------------|
| 1 | `1_clusters_and_markers.ipynb` | **[gpu]** Combine embryos, compute cell cycle scores and highly variable genes, cluster cells, and identify marker genes |
| 2 | `2_initial_claude_annotation.py` | Annotate initial clusters with Claude (Opus 4.5) based on marker genes and lineage-linked clusters |
| 3 | `3_refine_with_cellxgene.slurm` | Launch cellxgene instances per cluster for manual annotation refinement |
| 4 | `4_cleanup_and_markers.ipynb` | Drop doublets identified during annotation and identify cell subtype marker genes |
| 5 | `5_final_claude_annotation.py` | Generate descriptions and naming rationales for final cell subtypes with Claude (Table S2) |

Manually curated labels from step 3 are saved to `data/cell_types.csv`, which is read by step 4.

## Validation

Notebooks validating the PEtracer system and tree reconstruction accuracy. They use preliminary bulk and scRNA-seq data as well as the full scRNA-seq dataset. Steps should be run in order from the `validation/` directory.

| Step | Script | Description |
|------|--------|-------------|
| 1 | `1_ideogram.html` | Ideogram of PEtracer lineage tracing cassette genomic locations |
| 2 | `2_bulk_kinetics.ipynb` | Editing kinetics in vitro and in vivo from bulk sequencing, and lineage mark balance |
| 3 | `3_preliminary_tracing.ipynb` | Preliminary scRNA-seq tracing data for slow, intermediate, and fast kinetics lines: chimerism, edit fractions, and cell type distributions |
| 4 | `4_tracing_performance.ipynb` | Tracing performance in the full dataset: detection rate, chimerism, extant cells over time, fraction of branches marked, PEmax heritability, germ layer mismatch, and per-embryo statistics (Table S1) |

## Embedding

Scripts and notebooks for training the scVI model and computing UMAP embeddings. Run from the `embedding/` directory. Steps 1 and 2 require a GPU.

| Step | Script | Description |
|------|--------|-------------|
| 1 | `1_scVI_embedding.slurm` | **[gpu]** Train the scVI model and save latent embeddings to `data/scvi.csv` (runs `scVI_embedding.py`) |
| 2 | `2_umap_embedding.ipynb` | **[gpu]** Compute global and cluster UMAP embeddings |
| 3 | `3_plot_umaps.ipynb` | Plot UMAP embeddings colored by cell type, germ layer, stage, replicate, donor/host, and cell cycle phase |

## Staging

Scripts for integrating the dataset with published reference atlases and estimating transcriptional similarity between embryos. Run from the `staging/` directory. Steps 1–4 require the **[gpu]** environment.

Before running, download the reference datasets and place them in `data/external/`:

- [Pijuan-Sala et al. 2019](https://doi.org/10.1038/s41586-019-0933-9) gastrulation atlas (E6.5–E8.5), as processed by [Qiu et al. 2022](https://doi.org/10.1038/s41588-022-01018-x) and available from [TOME](https://tome.gs.washington.edu/), saved as `data/external/pijuan-sala2019.h5ad`
- [Qiu et al. 2024](https://doi.org/10.1038/s41586-024-07069-w) organogenesis atlas (E8.0–P0), available from [omg.gs.washington.edu](https://omg.gs.washington.edu/jax/public/about.html), saved as `data/external/qiu2025_1.h5ad`
- An Ensembl gene ID to MGI symbol table from [Ensembl BioMart](https://www.ensembl.org/biomart), saved as `data/external/gene_symbols.tsv`

| Step | Script | Description |
|------|--------|-------------|
| 1 | `1_pijuan-sala_integration.py` | **[gpu]** Integrate E7.5–E8.5 embryos with Pijuan-Sala et al. 2019 using scVI |
| 2 | `2_qiu_integration.py` | **[gpu]** Integrate E8.5–E10.0 embryos with Qiu et al. 2024 using scVI |
| 3 | `3_integration_ot_metrics.py` | **[gpu]** Sinkhorn similarity between our embryos and reference embryos in the integrated latent spaces |
| 4 | `4_embryo_ot_metrics.py` | **[gpu]** Sinkhorn similarity between our embryos, and between host and donor cells within each embryo |
| 5 | `5_staging_plots.ipynb` | Plot integrated UMAPs and transcriptional similarity heatmaps |

## Simulation

Scripts for in silico benchmarking of lineage tree reconstruction and branch-length estimation. Steps 1–3 run as Slurm job arrays from the `simulation/` directory. Step 4 plots the results.

[LAML-Pro](https://github.com/raphael-group/LAML-Pro) (`pylaml`) is not included in the conda environment and must be installed separately before running steps 1 and 3.

| Step | Script | Description |
|------|--------|-------------|
| 1 | `1_benchmark_solver.slurm` | Benchmark reconstruction accuracy, runtime, and memory of Neighbor Joining, UPGMA, Greedy, LAML-Pro, and FastTree across tree sizes (runs `benchmark_solver.py`) |
| 2 | `2_benchmark_capacity.slurm` | Benchmark FastTree reconstruction accuracy across number of cassettes and missing-data rate (runs `benchmark_capacity.py`) |
| 3 | `3_benchmark_ble.slurm` | Benchmark ConvexML and LAML-Pro branch-length estimation across number of cassettes and division-time variance (runs `benchmark_ble.py`) |
| 4 | `4_simulation_plots.ipynb` | Plot example simulated trees and benchmarking results |

## Stability

Scripts for assessing the robustness of lineage statistics to integration dropout and topological error. Run from the `stability/` directory.

| Step | Script | Description |
|------|--------|-------------|
| 1 | `1_jackknife.slurm` | Leave-one-integration-out jackknife reconstruction of the E9.5 trees, with clade statistics and germ layer linkage recomputed for each replicate (runs `jackknife.py`) |
| 2 | `2_permutation.slurm` | Simulate leaf misplacement on trees with known fate structure and measure the effect on clade statistics (runs `permutation.py`) |
| 3 | `3_stability_plots.ipynb` | Plot Robinson-Foulds distances, clade statistics, and linkage across jackknife and simulated-error replicates |

## Fate

Notebooks and scripts for plotting lineage trees and analyzing fate restriction. Steps should be run in order from the `fate/` directory.

**Software:** Claude Code CLI v2.1.138

| Step | Script | Description |
|------|--------|-------------|
| 1 | `1_plot_trees.ipynb` | Plot lineage trees colored by germ layer and cell type, example clades, and their UMAP embeddings |
| 2 | `2_ancestral_linkage.slurm` | Compute per-embryo ancestral linkage for germ layers, cell types, and cell subtypes (runs `ancestral_linkage.py`) |
| 3 | `3_annotate_linkage.py` | Annotate significant E9.5 cell type linkages with a rationale and novelty score using Claude (Table S3) |
| 4 | `4_plot_linkage.ipynb` | Plot ancestral linkage heatmaps for germ layers, cell types, and cell subtypes |
| 5 | `5_germ_layer_dynamics.ipynb` | Germ layer fate bias, fate-restricted clades, restriction timing, and restriction Sankey diagrams |
| 6 | `6_cell_type_dynamics.ipynb` | Number, size, and restriction timing of cell type and subtype fate-restricted clades (Table S4) |
| 7 | `7_bias_and_stereotypy.ipynb` | Early fate bias patterns of ancestral nodes, label-independent transcriptional distance, and reproducibility across replicates |
| 8 | `8_compute_node_mmd.slurm` | Per-node transcriptional distance (energy distance in scVI space) between descendants and all cells at the same stage, used in step 7 (runs `compute_node_mmd.py`; requires a GPU node) |

## Gene Programs

Notebooks and scripts for identifying and annotating heritable gene programs. Steps should be run in order from the `gene_programs/` directory.

**Software:** Claude Code CLI v2.1.138

| Step | Script | Description |
|------|--------|-------------|
| 1 | `1_define_programs.ipynb` | Identify heritable gene programs with Hotspot and score program activity in each cell |
| 2 | `2_annotate_programs.py` | Name and describe each program with Claude (Opus 4.6) (Table S5) |
| 3 | `3_analyze_programs.ipynb` | Visualize program activation across cell types, program rates of change along lineages, and transcriptional divergence |

## Spatial Patterning

Notebooks for inferring spatial positions and analyzing neural patterning. Run from the `spatial_patterning/` directory.

Before running step 1, download section E13.5_E1S3 of the [Chen et al. 2022](https://doi.org/10.1016/j.cell.2022.04.003) Stereo-seq (MOSTA) dataset from [STOmicsDB](https://db.cngb.org/stomics/datasets/STDS0000058/explore?section=E13.5_E1S3.MOSTA.h5ad). Place it in `data/external/` as `E13.5_E1S3.MOSTA.h5ad`.

Step 1 generates `results/spatial_scores.csv`. This file is also used by `vignettes/axial_progenitors.ipynb`. Instead of running step 1, you can download it from [Zenodo](https://doi.org/10.5281/zenodo.19892784) (`E7.5-E10_spatial_scores.csv`) and save it as `spatial_patterning/results/spatial_scores.csv`.

| Step | Script | Description |
|------|--------|-------------|
| 1 | `1_spatial_scores.ipynb` | Define A-P and D-V coordinates in the MOSTA reference, select spatially informative genes, and train random forest regressors to infer spatial scores for each cell |
| 2 | `2_neural_patterning.ipynb` | Neural-restricted clades along the A-P axis, spatial heritability, emergence of spatial bias, and midbrain-hindbrain boundary (r1) lineage separation |

## Vignettes

Notebooks for tissue-specific analyses. Each can be run independently from the `vignettes/` directory after the fate analyses. `axial_progenitors.ipynb` additionally requires `spatial_patterning/results/spatial_scores.csv`.

| Script | Description |
|--------|-------------|
| `notochord.ipynb` | Notochord fate restriction timing and sibling outputs relative to germ layer specification |
| `heart.ipynb` | Heart field lineage relationships, restriction timing, clade co-occurrence, and within-clade transcriptional similarity |
| `neural_crest.ipynb` | Trunk neural crest fate restriction and DRG/autonomic bipotency |
| `endothelial.ipynb` | Mesodermal origins of endothelial cells, their restriction timing, and residual transcriptional signatures of origin |
| `axial_progenitors.ipynb` | Neural-mesodermal sibling relationships along the A-P axis, axial progenitor bias and clonal persistence, and proliferation scores |

## Legends

| Script | Description |
|--------|-------------|
| `legends/plot.py` | Generate shared figure legends and colorbars |

## Citation

If you use this code or data, please cite:

> Colgan WN, Koblan LW, et al. Comprehensive Lineage Tracing Maps the Landscape of Cell Fate Decisions in Mouse Embryogenesis. 2026.

## License

This code is released under the BSD 3-Clause License (see [LICENSE](LICENSE)).
