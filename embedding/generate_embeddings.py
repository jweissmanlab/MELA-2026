import scvi
import scanpy as sc
import pandas as pd


ADATA_FILE = "/home/gokulg/orcd/scratch/lt/data/embryos/combined_counts.h5ad"
HVGS_FILE = "/home/gokulg/orcd/scratch/lt/data/hvg_v5.csv"
SCVI_FILE = "models/scvi/kl0.0_d50_l2/scvi_model"
SAVE_FILE = "/home/gokulg/orcd/scratch/lt/data/embryos/embryos_hvg_v5_scvi.h5ad"

adata = sc.read_h5ad(ADATA_FILE)
hvgs = pd.read_csv(HVGS_FILE, header=None)[0].tolist()
adata = adata[:, hvgs].copy()

model = scvi.model.SCVI.load(SCVI_FILE, adata)

embeddings = model.get_latent_representation()

adata.obsm["X_scvi"] = embeddings
adata.write_h5ad(SAVE_FILE)
