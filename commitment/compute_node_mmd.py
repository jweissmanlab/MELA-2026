import treedata
from geomloss import SamplesLoss
import numpy as np
import networkx as nx
from collections import defaultdict
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scanpy as sc
import pickle

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
n_subsample = 100      # max cells to subsample per comparison
n_resamples = 10       # number of subsamples to average over
n_perms = 100          # number of permutations for null distribution
threshold_fraction = 0.9

# Load data

adata = treedata.read_h5td(
    '../data/embryos/kl0.0_d50_l2_covnocov_adata_with_embeddings.h5ad'
)

adata.obs['time'] = [float(x.split('-')[0][1:]) for x in adata.obs['embryo'].values]


print('data loaded')

tree_filenames = []
for e in ["7.5", "8.0", "8.5", "9.0", "9.5"]:
    for r in [1, 2, 3]:
        tree_filenames.append(f"e{e}_r{r}_tree")
tree_filenames.append("e10.0_r1_tree")

tdatas = [
    treedata.read_h5td(f"../data/embryos/{f}.h5td")
    for f in tree_filenames
]

print('tree loaded')

trees, tree_names = [], []
for td in tdatas:
    for k in td.obst.keys():
        trees.append(td.obst[k])
        tree_names.append(k)

# Precompute lookups

name_to_idx = {name: idx for idx, name in enumerate(adata.obs_names)}
states = adata.obsm['X_scvi']
cell_times = adata.obs['time'].values  # time per cell

# Build per-timepoint index arrays
timepoint_indices = defaultdict(list)
for idx, t in enumerate(cell_times):
    timepoint_indices[t].append(idx)
timepoint_indices = {t: np.array(idxs) for t, idxs in timepoint_indices.items()}

# Precompute per-timepoint state tensors (full, for subsampling later)
timepoint_states = {t: states[idxs] for t, idxs in timepoint_indices.items()}

mmd_loss = SamplesLoss("energy")

# Collect all leaves across trees
all_leaves = []
for tree in trees:
    all_leaves.extend(
        [n for n in tree.nodes() if tree.out_degree(n) == 0 and n in name_to_idx]
    )

# Cache leaf descendants per node
descendants_cache = {}

def get_leaf_descendants(node, tree):
    if node in descendants_cache:
        return descendants_cache[node]
    if tree.out_degree(node) == 0:
        result = {node}
    else:
        result = {
            d for d in nx.descendants(tree, node)
            if tree.out_degree(d) == 0
        }
    descendants_cache[node] = result
    return result


def subsample_tensor(arr, n):
    """Subsample rows of a numpy array and return a float tensor."""
    if len(arr) > n:
        idx = np.random.choice(len(arr), n, replace=False)
        return torch.from_numpy(arr[idx]).float()
    return torch.from_numpy(arr).float()


# Compute MMD per node (timepoint-matched, averaged, with permutations)

node_mmd = {}  # node -> { 'mmd': float, 'mmd_perm_mean': float, 'mmd_perm_std': float, 'time': float }

for tree, tname in tqdm(zip(trees, tree_names), desc="processing trees", total=len(trees)):
    for node in tqdm(tree.nodes(), desc="MMD per node", leave=False):

        # ---- descendant leaves ----
        descendant_leaves = get_leaf_descendants(node, tree)
        descendant_indices = np.array(
            [name_to_idx[l] for l in descendant_leaves if l in name_to_idx]
        )
        if descendant_indices.size == 0:
            continue

        descendant_states_full = states[descendant_indices]
        n_desc = len(descendant_indices)

        # ---- node time ----
        node_time = tree.nodes[node].get('time', None)
        if node_time is None:
            continue

        # ---- reference: all cells at this timepoint ----
        ref_states_full = timepoint_states.get(node_time)
        if ref_states_full is None or len(ref_states_full) == 0:
            continue

        n_ref = len(ref_states_full)

        # ---- observed MMD (average over n_resamples) ----
        mmd_vals = []
        for _ in range(n_resamples):
            desc_t = subsample_tensor(descendant_states_full, n_subsample)
            ref_t = subsample_tensor(ref_states_full, n_subsample)
            mmd_vals.append(mmd_loss(ref_t, desc_t).item())
        obs_mmd = float(np.mean(mmd_vals))

        # ---- permutation null: sample random leaves from this timepoint ----
        # We sample the same number of cells as descendant_leaves from the
        # timepoint reference pool, then compare against the full timepoint pool.
        perm_mmds = []
        ref_indices_tp = timepoint_indices[node_time]

        for _ in range(n_perms):
            # draw n_desc cells without replacement from the timepoint pool
            n_draw = min(n_desc, len(ref_indices_tp))
            perm_idx = np.random.choice(len(ref_indices_tp), n_draw, replace=False)
            perm_states = ref_states_full[perm_idx]

            # average over resamples for this permutation too
            perm_resample_vals = []
            for _ in range(n_resamples):
                perm_t = subsample_tensor(perm_states, n_subsample)
                ref_t = subsample_tensor(ref_states_full, n_subsample)
                perm_resample_vals.append(mmd_loss(ref_t, perm_t).item())
            perm_mmds.append(float(np.mean(perm_resample_vals)))

        perm_mmds = np.array(perm_mmds)

        # empirical p-value: fraction of permutations >= observed
        p_value = float(np.mean(perm_mmds >= obs_mmd))

        node_mmd[node] = {
            'mmd': obs_mmd,
            'time': node_time,
            'perm_mean': float(np.mean(perm_mmds)),
            'perm_std': float(np.std(perm_mmds)),
            'p_value': p_value,
        }

# Aggregate per-leaf trajectories (root --> leaf)

# results = {}

# for tree in tqdm(trees, desc="aggregating trajectories"):
#     leaves_in_tree = [
#         n for n in tree.nodes()
#         if tree.out_degree(n) == 0 and n in name_to_idx
#     ]
#     for leaf in leaves_in_tree:
#         if leaf not in node_mmd:
#             continue

#         trajectory = [node_mmd[leaf]]
#         current = leaf
#         while list(tree.predecessors(current)):
#             parent = list(tree.predecessors(current))[0]
#             if parent in node_mmd:
#                 trajectory.append(node_mmd[parent])
#             current = parent

#         results[leaf] = trajectory


with open(
    "commitment/node_mmd_dictionary_with_perms.pkl", "wb"
) as f:
    pickle.dump(node_mmd, f)