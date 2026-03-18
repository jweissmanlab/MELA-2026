import sys
import os
import treedata
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import scvi
import seaborn as sns
import matplotlib.pyplot as plt
# from latentmi import lmi

# config from command line args
# usage: python eval_embeddings.py <kl_weight> <n_latent> <n_layers>
# pass "pca" as kl_weight to run pca baseline
mode = sys.argv[1]          # float kl_weight or "pca"
n_latent = int(sys.argv[2]) # latent dim
n_layers = int(sys.argv[3]) # number of hidden layers

# covariates
COVARIATE_KEYS = ["embryo", "type"]
DATA_DIR = "data/embryos"
HVG_FILE = f"{DATA_DIR}/hvg_2000.txt"


is_pca = (mode == "pca")
kl_weight = 0.0 if is_pca else float(mode)
tag = f"pca_d{n_latent}" if is_pca else f"kl{kl_weight}_d{n_latent}_l{n_layers}"
out_dir = f"models/scvi_covariates/{tag}"
os.makedirs(out_dir, exist_ok=True)

print(f"running config: {tag}")

# load data
adata = ad.read_h5ad(f"{DATA_DIR}/combined_counts.h5ad")
hvgs = pd.read_csv(HVG_FILE, header=None)[0].tolist()
adata = adata[:, hvgs].copy()

# compute embeddings
if is_pca:
    # preprocess
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=n_latent)
    rep_key = "X_pca"
else:
    scvi.model.SCVI.setup_anndata(adata,
                                  categorical_covariate_keys=COVARIATE_KEYS,)
    model = scvi.model.SCVI(
        adata,
        n_layers=n_layers,
        n_hidden=512,
        n_latent=n_latent,
        gene_likelihood="nb",
    )
    model.train(
        # max_epochs=100,
        max_epochs=15,
        early_stopping=True,
        early_stopping_patience=10,
        plan_kwargs={"kl_weight": kl_weight},
    )
    model.save(os.path.join(out_dir, "scvi_model"), overwrite=True)
    adata.obsm["X_scvi"] = model.get_latent_representation()
    rep_key = "X_scvi"

print(f"embeddings computed, shape: {adata.obsm[rep_key].shape}")

# time-mi on subsample
n_sub = min(500_000, adata.shape[0])
idx = np.random.choice(adata.shape[0], size=n_sub, replace=False)
adata_sub = adata[idx].copy()

# parse time from embryo name (e.g. "e7.5_r1" -> 7.5)
adata_sub.obs["time"] = adata_sub.obs["embryo"].str.extract(r"e(\d+\.?\d*)").astype(float).values

# # estimate time-mi on a 10k subsample
# n_mi_sub = min(10_000, adata_sub.shape[0])
# mi_idx = np.random.choice(adata_sub.shape[0], size=n_mi_sub, replace=False)
# adata_sub_sub = adata_sub[mi_idx].copy()

# pmis, _, _ = lmi.estimate(
#     adata_sub_sub.obsm[rep_key],
#     adata_sub_sub.obs["time"].values.reshape(-1, 1),
#     estimate_on_val=False,
# )

# time_mi = float(np.nanmean(pmis))
# print(f"time-mi: {time_mi:.4f}")

# tree-mi per embryo
# tree_names = []
# for e in ["7.5", "8.0", "8.5", "9.0", "9.5"]:
#     for r in [1, 2, 3]:
#         tree_names.append(f"e{e}_r{r}_tree")
# tree_names.append("e10.0_r1_tree")

# obs_index = adata.obs_names
# rep = adata.obsm[rep_key]

# rows = []
# for tdname in tree_names:
#     td_tree = treedata.read_h5td(f"{DATA_DIR}/{tdname}.h5td")
#     tree = td_tree.obst[td_tree.obs["tree"].value_counts().index[0]]

#     # collect sibling leaf pairs
#     sibling_pairs = []
#     for node in tree.nodes():
#         children = list(tree.successors(node))
#         leaves = [c for c in children if tree.out_degree(c) == 0]
#         for i in range(len(leaves)):
#             for j in range(i + 1, len(leaves)):
#                 sibling_pairs.append((leaves[i], leaves[j]))
    
#     # shuffle sibling pairs
#     np.random.shuffle(sibling_pairs)

#     # gather embedding vectors for siblings present in adata
#     a_states, b_states = [], []
#     for sa, sb in sibling_pairs[:10_000]:
#         if sa in obs_index and sb in obs_index:
#             a_states.append(rep[obs_index.get_loc(sa)])
#             b_states.append(rep[obs_index.get_loc(sb)])

#     if len(a_states) < 50:
#         print(f"{tdname}: too few pairs ({len(a_states)}), skipping")
#         rows.append({"tree": tdname, "tree_mi": np.nan, "n_pairs": len(a_states)})
#         continue

#     pmis, _, _ = lmi.estimate(
#         np.array(a_states), np.array(b_states), estimate_on_val=False
#     )
#     tree_mi = float(np.nanmean(pmis))
#     print(f"{tdname}: tree-mi={tree_mi:.4f} (n={len(a_states)})")
#     rows.append({"tree": tdname, "tree_mi": tree_mi, "n_pairs": len(a_states)})

# save results
df = pd.DataFrame(rows)
# df["time_mi"] = time_mi
df["tag"] = tag
df["kl_weight"] = kl_weight if not is_pca else np.nan
df["n_latent"] = n_latent
df["n_layers"] = n_layers if not is_pca else np.nan
df["mode"] = "pca" if is_pca else "scvi"
df.to_csv(os.path.join(out_dir, "results.csv"), index=False)
print(f"results saved to {out_dir}/results.csv")

# umap on subsample
sc.pp.neighbors(adata_sub, use_rep=rep_key)
sc.tl.umap(adata_sub)

# palette: 3 shades per timepoint + darkkhaki for e10
palette = []
for cmap in ["Blues", "Oranges", "Greens", "Purples", "Greys"]:
    palette += sns.color_palette(cmap, 4)[-3:]
palette += ["darkkhaki"]

fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(
    x=adata_sub.obsm["X_umap"][:, 0],
    y=adata_sub.obsm["X_umap"][:, 1],
    hue=adata_sub.obs["embryo"],
    s=2, alpha=0.3, palette=palette, ax=ax,
)
ax.set(xticks=[], yticks=[], title=tag)
leg = ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", markerscale=3)

# Source - https://stackoverflow.com/a/42403471
# Posted by lhuber, modified by community. See post 'Timeline' for change history
# Retrieved 2026-02-05, License - CC BY-SA 4.0
for lh in leg.legend_handles:
    lh.set_alpha(1)

fig.savefig(os.path.join(out_dir, "umap.png"), dpi=200, bbox_inches="tight")
print(f"umap saved to {out_dir}/umap.png")