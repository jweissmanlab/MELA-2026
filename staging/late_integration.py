import scanpy as sc
import anndata as ad
import pandas as pd
import scvi
import numpy as np
import os

DATA_DIR = "/mnt/home/gokulg/data/"

adata_qiu24 = sc.read_h5ad("/mnt/home/gokulg/data/qiu2024/qiu2024_1.h5ad")
print('loaded qiu2024_1')
adata_devmap = sc.read_h5ad("/mnt/home/gokulg/data/embryos/combined_counts.h5ad")
print('loaded devmap')
OUT_DIR = "models/integrated_scvi/"
ANNOTATION_FILE = os.path.join(DATA_DIR, "embryos/annotation_v5.csv")
TYPES_TO_REMOVE = ["doublet"]

qiu24_overlapping_days = ['E8.5', 'E8.75', 'E9.0', 'E9.5', 'E9.25', 'E9.75', 'E10.0']

hvgs = pd.read_csv(f"{DATA_DIR}embryos/hvg_v5.csv", header=None)[0].tolist()
gene_symbols = pd.read_csv(f"{DATA_DIR}gene_symbols.tsv", sep='\t', header=0)

conversion_dict = {}
for i, row in gene_symbols.iterrows():
    conversion_dict[row['Gene stable ID']] = row['MGI symbol']
    conversion_dict[row['Gene stable ID'].split('.')[0]] = row['MGI symbol']

# convert var_names for qiu24
adata_qiu24.var_names = [str(conversion_dict.get(v, v)) for v in adata_qiu24.var_names]
adata_qiu24.var_names_make_unique()

# find common HVGs across qiu24 and devmap
common_hvgs = [h for h in hvgs if h in adata_qiu24.var_names and h in adata_devmap.var_names]
print(f"Common HVGs across qiu24 + devmap: {len(common_hvgs)} / {len(hvgs)}")

# subset to common HVGs
adata_qiu24_hvg = adata_qiu24[:, common_hvgs].copy()
adata_devmap_hvg = adata_devmap[:, common_hvgs].copy()

# add annotations and remove unwanted types from devmap
adata_devmap_hvg.obs['time'] = [float(x.split('-')[0][1:]) for x in adata_devmap_hvg.obs['embryo'].values]
annotations = pd.read_csv(ANNOTATION_FILE, index_col=0)
adata_devmap_hvg.obs = adata_devmap_hvg.obs.join(annotations, how="left")
adata_devmap_hvg = adata_devmap_hvg[~adata_devmap_hvg.obs["annotation"].isin(TYPES_TO_REMOVE)].copy()

# filter devmap to late timepoints only
adata_devmap_hvg = adata_devmap_hvg[adata_devmap_hvg.obs['time'] >= 8.49].copy()

# filter qiu24 to overlapping days
adata_qiu24_hvg = adata_qiu24_hvg[adata_qiu24_hvg.obs['day'].isin(qiu24_overlapping_days)].copy()

# check integer counts
assert (adata_qiu24_hvg.X.data[:10].astype(int) == adata_qiu24_hvg.X.data[:10]).all()
assert (adata_devmap_hvg.X.data[:10].astype(int) == adata_devmap_hvg.X.data[:10]).all()

print(f"Qiu2024 shape after filtering: {adata_qiu24_hvg.shape}")
print(f"devmap (late) shape after filtering: {adata_devmap_hvg.shape}")

# concatenate
combined = ad.concat(
    [adata_qiu24_hvg, adata_devmap_hvg],
    axis='obs', join='outer', label='dataset',
    keys=['qiu2024', 'devmap']
)
print(f"Combined (late) shape: {combined.shape}")
# save counts in layers counts
combined.layers["counts"] = combined.X.copy()
sc.pp.normalize_total(combined, target_sum=1e4)
sc.pp.log1p(combined)
# compute DEGs between devmap and qiu2024, then remove from combined
k = 250
n_sub = min(100_000, combined.shape[0])
n_per_ds = n_sub // combined.obs['dataset'].nunique()
idx = np.concatenate([
    np.random.choice(np.where(combined.obs['dataset'] == d)[0],
                     size=min(n_per_ds, (combined.obs['dataset'] == d).sum()), replace=False)
    for d in combined.obs['dataset'].unique()
])

# save counts in layers counts
combined.layers["counts"] = combined.X.copy()
sc.pp.normalize_total(combined, target_sum=1e4)
sc.pp.log1p(combined)

combined_sub = combined[idx].copy()
sc.tl.rank_genes_groups(combined_sub, 'dataset', method='wilcoxon')
deg_df = sc.get.rank_genes_groups_df(combined_sub, group=None, key='rank_genes_groups')
deg_names = set(deg_df.groupby('group').head(k)['names'].values)

gene_subset = [x for x in combined.var_names if x not in deg_names]
print(f"Removing {len(deg_names)} DEGs, keeping {len(gene_subset)} genes")
combined = combined[:, gene_subset].copy()

# scVI setup and training
scvi.model.SCVI.setup_anndata(combined, layer='counts', categorical_covariate_keys=['dataset'])

model = scvi.model.SCVI(
    combined,
    n_layers=2,
    n_latent=50,
    n_hidden=512,
    gene_likelihood='nb',
    dispersion='gene-batch'
)

model.train(max_epochs=20, early_stopping=False)

model.save(os.path.join(OUT_DIR, "scvi_model_late"), overwrite=True)
combined.obsm["X_scvi"] = model.get_latent_representation()
print(f"Late embeddings computed, shape: {combined.obsm['X_scvi'].shape}")

# subsample for umap
n_sub = min(500_000, combined.shape[0])
idx = np.random.choice(combined.shape[0], size=n_sub, replace=False)
combined_sub = combined[idx].copy()

sc.pp.neighbors(combined_sub, use_rep="X_scvi")
sc.tl.umap(combined_sub)

sc.pl.umap(combined_sub, color=["day"], size=2, hspace=0.6, save=True)
sc.pl.umap(combined_sub, color=["time"], size=2, hspace=0.6, save=True)

combined.write_h5ad(os.path.join(DATA_DIR, "qiu24_late_integrated_scvi.h5ad"))
print("Late integration saved.")