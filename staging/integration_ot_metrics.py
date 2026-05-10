import os
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from geomloss import SamplesLoss
from tqdm.auto import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product as iterproduct

DATA_DIR = "/mnt/home/gokulg/data/"
BUCKETS = {
    "early": os.path.join(DATA_DIR, "qiu22_early_integrated_scvi.h5ad"),
    "late": os.path.join(DATA_DIR, "qiu24_late_integrated_scvi.h5ad"),
}

OBSM_KEY = "X_scvi"
EMBRYO_COL_DEVMAP = "embryo"
EMBRYO_COL_QIU = "day"
TIME_COL_QIU = "day"
TIME_COL_DEVMAP = "time"
DATASET_COL = "dataset"

BATCH_SIZE = 512
N_BATCHES = 50

METRICS = {
    "sinkhorn": SamplesLoss("sinkhorn", p=2, blur=0.05),
    # "energy": SamplesLoss("energy", p=2),
}

def extract_embryo_data(adata):
    """Return {embryo_id: tensor} and {embryo_id: time}, grouped by dataset."""
    datasets = adata.obs[DATASET_COL].unique()
    print(datasets)
    print(adata.obs.keys())
    assert len(datasets) == 2, (
        f"Expected exactly 2 datasets within each bucket, got {list(datasets)}"
    )

    per_dataset = {}
    for ds in datasets:
        sub = adata[adata.obs[DATASET_COL] == ds]
        embryo_ids = sub.obs[EMBRYO_COL_DEVMAP if ds == "devmap" else EMBRYO_COL_QIU].unique()
        # drop nans if present
        embryo_ids = [e for e in embryo_ids if pd.notna(e)]
        print(embryo_ids)

        cells = {}
        times = {}
        for emb in embryo_ids:
            mask = sub.obs[EMBRYO_COL_DEVMAP if ds == "devmap" else EMBRYO_COL_QIU] == emb
            print(f"Dataset {ds} - embryo {emb}: {mask.sum()} cells")
            cells[emb] = torch.tensor(sub[mask].obsm[OBSM_KEY], dtype=torch.float32)
            times[emb] = sub.obs.loc[mask, TIME_COL_QIU if ds != "devmap" else TIME_COL_DEVMAP].iloc[0]

        order = sorted(embryo_ids, key=lambda e: times[e])
        per_dataset[ds] = {"cells": cells, "times": times, "order": order}

    return per_dataset


def compute_cross_dataset_ot(data_a, data_b, bucket_name, ds_a_name, ds_b_name):
    """Compute OT between every embryo in data_a x every embryo in data_b."""
    records = []
    pairs = list(iterproduct(data_a["order"], data_b["order"]))

    for emb_a, emb_b in tqdm(pairs, desc=f"cross-dataset ({bucket_name})"):
        cells_a = data_a["cells"][emb_a]
        cells_b = data_b["cells"][emb_b]
        n_a, n_b = cells_a.shape[0], cells_b.shape[0]

        if n_a == 0 or n_b == 0:
            continue

        eff_batches = min(N_BATCHES, n_a, n_b)
        if eff_batches == 0:
            continue

        perm_a = torch.randperm(n_a)
        perm_b = torch.randperm(n_b)
        parts_a = torch.tensor_split(perm_a, eff_batches)
        parts_b = torch.tensor_split(perm_b, eff_batches)

        for batch_idx in range(eff_batches):
            pool_a = cells_a[parts_a[batch_idx]]
            pool_b = cells_b[parts_b[batch_idx]]

            sa = pool_a[torch.randint(pool_a.shape[0], (BATCH_SIZE,))]
            sb = pool_b[torch.randint(pool_b.shape[0], (BATCH_SIZE,))]

            row = {
                "bucket": bucket_name,
                "dataset_a": ds_a_name,
                "dataset_b": ds_b_name,
                "embryo_a": emb_a,
                "embryo_b": emb_b,
                "batch": batch_idx,
                "n_a": n_a,
                "n_b": n_b,
                "time_a": data_a["times"][emb_a],
                "time_b": data_b["times"][emb_b],
            }
            for name, metric in METRICS.items():
                row[name] = metric(sa, sb).item()

            records.append(row)

    return pd.DataFrame(records)


def to_rectangular_matrix(df_pw, value_col, row_order, col_order):
    """Aggregate batches to mean, show as rectangular matrix."""
    agg = df_pw.groupby(["embryo_a", "embryo_b"])[value_col].mean()
    mat = pd.DataFrame(np.nan, index=row_order, columns=col_order)
    for (a, b), v in agg.items():
        mat.loc[a, b] = v
    return mat


all_records = []

for bucket_name, path in BUCKETS.items():
    print(f"\nLoading {bucket_name}: {path}")
    adata = sc.read_h5ad(path)

    per_dataset = extract_embryo_data(adata)
    del adata

    ds_names = list(per_dataset.keys())
    ds_a_name, ds_b_name = ds_names[0], ds_names[1]
    data_a, data_b = per_dataset[ds_a_name], per_dataset[ds_b_name]

    print(f"  {ds_a_name}: {len(data_a['order'])} embryos  by  "
          f"{ds_b_name}: {len(data_b['order'])} embryos")

    df_pw = compute_cross_dataset_ot(data_a, data_b, bucket_name, ds_a_name, ds_b_name)
    all_records.append(df_pw)

    for metric_name in METRICS:
        mat = to_rectangular_matrix(
            df_pw, metric_name, data_a["order"], data_b["order"]
        )
        row_labels = [f"{e} (t={data_a['times'][e]})" for e in data_a["order"]]
        col_labels = [f"{e} (t={data_b['times'][e]})" for e in data_b["order"]]

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            mat.astype(float),
            xticklabels=col_labels,
            yticklabels=row_labels,
            ax=ax,
            cmap="viridis",
        )
        ax.set_title(f"[{bucket_name}] {metric_name} distance\n"
                      f"rows={ds_a_name}  cols={ds_b_name}")
        ax.set_xlabel(ds_b_name)
        ax.set_ylabel(ds_a_name)
        plt.xticks(rotation=90, fontsize=7)
        plt.yticks(fontsize=7)
        plt.tight_layout()
        plt.savefig(f"embryo_cross_{bucket_name}_{metric_name}.png", dpi=200)
        plt.show()

        mat.to_csv(f"embryo_cross_{bucket_name}_{metric_name}_matrix.csv")
        print(f"  Saved matrix & plot: {bucket_name} / {metric_name}")

# Save combined raw records
df_all = pd.concat(all_records, ignore_index=True)
df_all.to_csv("pairwise_embryo_cross_dataset_distances.csv", index=False)
print(f"\nSaved {len(df_all)} total rows --> pairwise_embryo_cross_dataset_distances.csv")