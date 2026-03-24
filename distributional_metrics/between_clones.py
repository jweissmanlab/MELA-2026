from geomloss import SamplesLoss
import torch
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm.auto import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations

DATAFILE = "/home/gokulg/orcd/scratch/lt/data/kl0.0_d50_l2_covnocov_adata_with_embeddings.h5ad"
TYPES_TO_REMOVE = ["doublet"]

adata = sc.read_h5ad(DATAFILE)
adata = adata[~adata.obs["annotation"].isin(TYPES_TO_REMOVE)].copy()

obsm_key = "X_scvi"
batch_size = 512
n_batches = 50

metrics = {
    "sinkhorn": SamplesLoss("sinkhorn", p=2, blur=0.05),
    "energy": SamplesLoss("energy", p=2),
}

# ---- pre-extract cells per clone ----
clone_ids = adata.obs["clone"].unique()
clone_cells = {}

for clone in clone_ids:
    mask = adata.obs["clone"] == clone
    cells = torch.tensor(adata[mask].obsm[obsm_key], dtype=torch.float32)
    clone_cells[clone] = cells

# ---- pairwise: all clone pairs (upper triangle only) ----
pairwise_records = []
clone_list = list(clone_ids)

for i, j in tqdm(list(combinations(range(len(clone_list)), 2)), desc="pairwise clones"):
    cl_a, cl_b = clone_list[i], clone_list[j]
    cells_a = clone_cells[cl_a]
    cells_b = clone_cells[cl_b]
    n_a = cells_a.shape[0]
    n_b = cells_b.shape[0]

    if n_a == 0 or n_b == 0:
        continue

    eff_batches = min(n_batches, n_a, n_b)
    if eff_batches == 0:
        continue

    perm_a = torch.randperm(n_a)
    perm_b = torch.randperm(n_b)
    parts_a = torch.tensor_split(perm_a, eff_batches)
    parts_b = torch.tensor_split(perm_b, eff_batches)

    for batch_idx in range(eff_batches):
        pool_a = cells_a[parts_a[batch_idx]]
        pool_b = cells_b[parts_b[batch_idx]]

        sa = pool_a[torch.randint(pool_a.shape[0], (batch_size,))]
        sb = pool_b[torch.randint(pool_b.shape[0], (batch_size,))]

        row = {
            "clone_a": cl_a,
            "clone_b": cl_b,
            "batch": batch_idx,
            "n_a": n_a,
            "n_b": n_b,
        }
        for name, metric in metrics.items():
            row[name] = metric(sa, sb).item()
        pairwise_records.append(row)

df_pairwise = pd.DataFrame(pairwise_records)

# ---- pivot to symmetric matrix for heatmapping ----
def to_symmetric_matrix(df_pw, value_col, order):
    """Aggregate batches to mean, then fill both triangles."""
    agg = df_pw.groupby(["clone_a", "clone_b"])[value_col].mean()
    idx = {e: k for k, e in enumerate(order)}
    mat = pd.DataFrame(np.nan, index=order, columns=order)
    for (a, b), v in agg.items():
        mat.iloc[idx[a], idx[b]] = v
        mat.iloc[idx[b], idx[a]] = v
    np.fill_diagonal(mat.values, 0.0)
    return mat

for name in metrics:
    mat = to_symmetric_matrix(df_pairwise, name, clone_list)
    # fig, ax = plt.subplots(figsize=(8, 7))
    # sns.heatmap(mat.astype(float), annot=True, fmt=".3f", ax=ax, cmap="viridis")
    # ax.set_title(f"Pairwise {name} distance (clones)")
    # plt.tight_layout()
    # plt.show()

print(df_pairwise)

df_pairwise.to_csv("pairwise_clone_distributional_distances.csv", index=False)