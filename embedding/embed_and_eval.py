import sys
import os
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import scvi

# config from command line args: <kl_weight> <n_latent> <covariate_keys: "type"|"none">
kl_weight = float(sys.argv[1])
n_latent = int(sys.argv[2])
cov_arg = sys.argv[3]
COVARIATE_KEYS = [cov_arg] if cov_arg != "none" else []
N_LAYERS = 2
DATA_DIR = "../lt/data"
HVG_FILE = f"{DATA_DIR}/hvg_v5.csv"
ANNOTATION_FILE = f"{DATA_DIR}/annotation_v5.csv"
TYPES_TO_REMOVE = ["doublet"]

cov_tag = cov_arg if cov_arg != "none" else "nocov"
tag = f"kl{kl_weight}_d{n_latent}_l{N_LAYERS}_cov{cov_tag}"
out_dir = f"models/scvi/{tag}"
os.makedirs(out_dir, exist_ok=True)
print(f"running config: {tag}")

# load data and subset to hvgs
print("loading data...")
adata = ad.read_h5ad(f"{DATA_DIR}/embryos/combined_counts.h5ad")
hvgs = pd.read_csv(HVG_FILE, header=None)[0].tolist()
adata = adata[:, hvgs].copy()

# add annotations and remove unwanted types
annotations = pd.read_csv(ANNOTATION_FILE, index_col=0)
adata.obs = adata.obs.join(annotations, how="left")
adata = adata[~adata.obs["annotation"].isin(TYPES_TO_REMOVE)].copy()

# train scvi model
setup_kwargs = {}
if COVARIATE_KEYS:
    setup_kwargs["categorical_covariate_keys"] = COVARIATE_KEYS
scvi.model.SCVI.setup_anndata(adata, **setup_kwargs)
model = scvi.model.SCVI(
    adata, n_layers=N_LAYERS, n_hidden=512, n_latent=n_latent, gene_likelihood="nb",
)
model.train(
    max_epochs=15, early_stopping=True, early_stopping_patience=10,
    plan_kwargs={"kl_weight": kl_weight},
)
model.save(os.path.join(out_dir, "scvi_model"), overwrite=True)
adata.obsm["X_scvi"] = model.get_latent_representation()
print(f"embeddings computed, shape: {adata.obsm['X_scvi'].shape}")

adata.write_h5ad(os.path.join(DATA_DIR, f"{tag}_adata_with_embeddings.h5ad"))
print(f"adata with embeddings saved to {out_dir}/{tag}_adata_with_embeddings.h5ad")

# subsample for umap
n_sub = min(500_000, adata.shape[0])
idx = np.random.choice(adata.shape[0], size=n_sub, replace=False)
adata_sub = adata[idx].copy()

# compute and save umaps
sc.pp.neighbors(adata_sub, use_rep="X_scvi")
sc.tl.umap(adata_sub)
sc.pl.umap(adata_sub, color="embryo", size=2, title=tag, show=False,
           save=f"_{tag}_embryo.png")
sc.pl.umap(adata_sub, color="annotation", size=2, title=tag, show=False,
           save=f"_{tag}_annotation.png")
print(f"umaps saved to figures/")