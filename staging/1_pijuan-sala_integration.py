import os

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import treedata as td

DATA_DIR = "../data/"

adata_qiu22 = sc.read_h5ad(f"{DATA_DIR}external/pijuan-sala2019.h5ad")
print("loaded qiu2022")
adata_devmap = td.read_h5td(f"{DATA_DIR}counts.h5td")
print("loaded devmap")
OUT_DIR = "models/integrated_scvi/"
ANNOTATION_FILE = os.path.join(DATA_DIR, "obs.csv")
TYPES_TO_REMOVE = ["doublet"]

qiu22_overlapping_days = ["E6.5", "E6.75", "E7", "E7.25", "E7.5", "E7.75", "E8", "E8.25", "E8.5a"]

hvgs = pd.read_csv(f"{DATA_DIR}hvg.csv", header=None)[0].tolist()
gene_symbols = pd.read_csv(f"{DATA_DIR}external/gene_symbols.tsv", sep="\t", header=0)

conversion_dict = {}
for _, row in gene_symbols.iterrows():
    conversion_dict[row["Gene stable ID"]] = row["MGI symbol"]
    conversion_dict[row["Gene stable ID"].split(".")[0]] = row["MGI symbol"]

# convert var_names for qiu22
adata_qiu22.var_names = [str(conversion_dict.get(v, v)) for v in adata_qiu22.var_names]
adata_qiu22.var_names_make_unique()

# find common HVGs across qiu22 and devmap
common_hvgs = [h for h in hvgs if h in adata_qiu22.var_names and h in adata_devmap.var_names]
print(f"Common HVGs across qiu22 + devmap: {len(common_hvgs)} / {len(hvgs)}")

# subset to common HVGs
adata_qiu22_hvg = adata_qiu22[:, common_hvgs].copy()
adata_devmap_hvg = adata_devmap[:, common_hvgs].copy()

# add annotations and remove unwanted types from devmap
adata_devmap_hvg.obs["time"] = [float(x.split("-")[0][1:]) for x in adata_devmap_hvg.obs["embryo"].values]
annotations = pd.read_csv(ANNOTATION_FILE, index_col=0)[["cell_subtype"]].rename(columns={"cell_subtype": "annotation"})
adata_devmap_hvg.obs = adata_devmap_hvg.obs.join(annotations, how="left")
adata_devmap_hvg = adata_devmap_hvg[~adata_devmap_hvg.obs["annotation"].isin(TYPES_TO_REMOVE)].copy()

# filter devmap to early timepoints only
adata_devmap_hvg = adata_devmap_hvg[adata_devmap_hvg.obs["time"] <= 8.51].copy()

# filter qiu22 to overlapping days
adata_qiu22_hvg = adata_qiu22_hvg[adata_qiu22_hvg.obs["day"].isin(qiu22_overlapping_days)].copy()

# check integer counts
assert (adata_qiu22_hvg.X.data[:10].astype(int) == adata_qiu22_hvg.X.data[:10]).all()
assert (adata_devmap_hvg.X.data[:10].astype(int) == adata_devmap_hvg.X.data[:10]).all()

print(f"Qiu2022 shape after filtering: {adata_qiu22_hvg.shape}")
print(f"devmap (early) shape after filtering: {adata_devmap_hvg.shape}")

# concatenate
combined = ad.concat(
    [adata_qiu22_hvg, adata_devmap_hvg], axis="obs", join="outer", label="dataset", keys=["qiu2022", "devmap"]
)
print(f"Combined (early) shape: {combined.shape}")

# save counts in layers counts
combined.layers["counts"] = combined.X.copy()
sc.pp.normalize_total(combined, target_sum=1e4)
sc.pp.log1p(combined)

# compute DEGs between devmap and qiu2022, then remove from combined
k = 250
n_sub = min(100_000, combined.shape[0])
n_per_ds = n_sub // combined.obs["dataset"].nunique()
idx = np.concatenate(
    [
        np.random.choice(
            np.where(combined.obs["dataset"] == d)[0],
            size=min(n_per_ds, (combined.obs["dataset"] == d).sum()),
            replace=False,
        )
        for d in combined.obs["dataset"].unique()
    ]
)
combined_sub = combined[idx].copy()
sc.tl.rank_genes_groups(combined_sub, "dataset", method="wilcoxon")
deg_df = sc.get.rank_genes_groups_df(combined_sub, group=None, key="rank_genes_groups")
deg_names = set(deg_df.groupby("group").head(k)["names"].values)

gene_subset = [x for x in combined.var_names if x not in deg_names]
print(f"Removing {len(deg_names)} DEGs, keeping {len(gene_subset)} genes")
combined = combined[:, gene_subset].copy()

# scVI setup and training
scvi.model.SCVI.setup_anndata(combined, layer="counts", categorical_covariate_keys=["dataset"])

model = scvi.model.SCVI(combined, n_layers=2, n_latent=50, n_hidden=512, gene_likelihood="nb", dispersion="gene-batch")

model.train(max_epochs=10, early_stopping=False)

model.save(os.path.join(OUT_DIR, "scvi_model_early"), overwrite=True)
combined.obsm["X_scvi"] = model.get_latent_representation()
print(f"Early embeddings computed, shape: {combined.obsm['X_scvi'].shape}")

# subsample for umap
n_sub = min(500_000, combined.shape[0])
idx = np.random.choice(combined.shape[0], size=n_sub, replace=False)
combined_sub = combined[idx].copy()

sc.pp.neighbors(combined_sub, use_rep="X_scvi")
sc.tl.umap(combined_sub)

sc.pl.umap(combined_sub, color=["day"], size=2, hspace=0.6, save=True)
sc.pl.umap(combined_sub, color=["time"], size=2, hspace=0.6, save=True)

combined.write_h5ad(os.path.join(DATA_DIR, "pijuan-sala19_integrated_scvi.h5ad"))
print("early integration saved.")
