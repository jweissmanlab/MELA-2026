#!/usr/bin/env python
"""
Compute per-node MMD (energy distance) for a SINGLE tree, identified by
--tree-index.  Designed to be run as part of a SLURM job array.

Usage:
    python compute_node_mmd.py \
        --adata ../lt/data/kl0.0_d50_l2_covnocov_adata_with_embeddings.h5ad \
        --tree-dir ../lt/data/embryos \
        --outdir commitment/per_tree \
        --tree-index 0 \
        --n-subsample 200 \
        --n-resamples 5 \
        --n-perms 10 \
        --min-descendants 5
"""

import argparse
import os
import pickle
from collections import defaultdict

import numpy as np
import networkx as nx
import torch
import treedata
from geomloss import SamplesLoss
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adata", required=True)
    p.add_argument("--tree-dir", required=True)
    p.add_argument("--outdir", required=True,
                    help="Directory for per-tree output pickles")
    p.add_argument("--tree-index", type=int, required=True,
                    help="Which tree to process (0-indexed into the tree list)")
    p.add_argument("--n-subsample", type=int, default=200)
    p.add_argument("--n-resamples", type=int, default=5)
    p.add_argument("--n-perms", type=int, default=10)
    p.add_argument("--min-descendants", type=int, default=5)
    p.add_argument("--device", default=None)
    return p.parse_args()


def subsample_gpu(tensor, n):
    if tensor.shape[0] > n:
        idx = torch.randperm(tensor.shape[0], device=tensor.device)[:n]
        return tensor[idx]
    return tensor


def get_leaf_descendants(node, tree, cache):
    if node in cache:
        return cache[node]
    if tree.out_degree(node) == 0:
        result = {node}
    else:
        result = {d for d in nx.descendants(tree, node)
                  if tree.out_degree(d) == 0}
    cache[node] = result
    return result


# ---------------------------------------------------------------------------
# Build the ordered list of (tree, name) pairs — must match across all jobs
# ---------------------------------------------------------------------------

def load_all_trees(tree_dir):
    tree_filenames = []
    for e in ["7.5", "8.0", "8.5", "9.0", "9.5"]:
        for r in [1, 2, 3]:
            tree_filenames.append(f"e{e}_r{r}_tree")
    tree_filenames.append("e10.0_r1_tree")

    tdatas = [
        treedata.read_h5td(f"{tree_dir}/{f}.h5td")
        for f in tree_filenames
    ]

    trees, tree_names = [], []
    for td in tdatas:
        for k in td.obst.keys():
            trees.append(td.obst[k])
            tree_names.append(k)
    return trees, tree_names


# ---------------------------------------------------------------------------
# Precomputation (only for the one tree we need)
# ---------------------------------------------------------------------------

def precompute(adata, tree, device):
    name_to_idx = {name: i for i, name in enumerate(adata.obs_names)}
    states = adata.obsm['X_scvi']
    cell_times = adata.obs['time'].values

    # Per-timepoint GPU tensors (reference populations)
    tp_indices = defaultdict(list)
    for i, t in enumerate(cell_times):
        tp_indices[t].append(i)
    tp_indices = {t: np.array(v) for t, v in tp_indices.items()}

    tp_states_np = {t: states[idx] for t, idx in tp_indices.items()}
    tp_tensors = {
        t: torch.from_numpy(arr).float().to(device)
        for t, arr in tp_states_np.items()
    }

    # Descendant info for every node in this tree.
    # Derive the relevant timepoint from the leaves themselves.
    desc_cache = {}
    node_info = {}
    for node in tree.nodes():
        leaves = get_leaf_descendants(node, tree, desc_cache)
        idxs = [name_to_idx[l] for l in leaves if l in name_to_idx]
        if not idxs:
            continue
        idx_arr = np.array(idxs)

        # All leaves share one timepoint
        t = float(np.round(cell_times[idx_arr[0]], 2))
        if t not in tp_tensors:
            continue

        node_info[node] = {
            'tensor': torch.from_numpy(states[idx_arr]).float().to(device),
            'leaf time': t,
            'node time' : tree.nodes[node]['time'],
            'n_desc': len(idx_arr),
        }

    return tp_indices, tp_tensors, node_info


# ---------------------------------------------------------------------------
# MMD computation for one tree
# ---------------------------------------------------------------------------

def compute_mmd(args, tree, tp_tensors, node_info, device):
    mmd_loss = SamplesLoss("energy")
    node_mmd = {}

    eligible = [n for n in tree.nodes()
                if n in node_info and node_info[n]['n_desc'] >= args.min_descendants]

    for node in tqdm(eligible, desc="nodes"):
        info = node_info[node]
        desc_tensor = info['tensor']
        ref_tensor = tp_tensors[info['leaf time']]
        n_desc = info['n_desc']

        # Observed MMD
        mmd_vals = torch.empty(args.n_resamples)
        for r in range(args.n_resamples):
            ref_t = subsample_gpu(ref_tensor, args.n_subsample)
            desc_t = subsample_gpu(desc_tensor, args.n_subsample)
            mmd_vals[r] = mmd_loss(ref_t, desc_t)
        obs_mmd = mmd_vals.mean().item()

        # Permutation null
        ref_n = ref_tensor.shape[0]
        n_draw = min(n_desc, ref_n)
        perm_mmds = torch.empty(args.n_perms)

        for p in range(args.n_perms):
            perm_idx = torch.randperm(ref_n, device=device)[:n_draw]
            perm_tensor = ref_tensor[perm_idx]

            perm_vals = torch.empty(args.n_resamples)
            for r in range(args.n_resamples):
                perm_t = subsample_gpu(perm_tensor, args.n_subsample)
                ref_t = subsample_gpu(ref_tensor, args.n_subsample)
                perm_vals[r] = mmd_loss(ref_t, perm_t)
            perm_mmds[p] = perm_vals.mean()

        perm_np = perm_mmds.cpu().numpy()

        node_mmd[node] = {
            'mmd': obs_mmd,
            'leaf time': info['leaf time'],
            'node time': info['node time'],
            'perm_mean': float(perm_np.mean()),
            'perm_std': float(perm_np.std()),
            'p_value': float(np.mean(perm_np >= obs_mmd)),
            'n_desc': n_desc,
        }

    return node_mmd


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    trees, tree_names = load_all_trees(args.tree_dir)
    n_trees = len(trees)
    print(f"Total trees: {n_trees}, processing index {args.tree_index}")

    if args.tree_index < 0 or args.tree_index >= n_trees:
        raise ValueError(f"--tree-index {args.tree_index} out of range [0, {n_trees})")

    tree = trees[args.tree_index]
    tname = tree_names[args.tree_index]
    print(f"Tree: {tname}  |  Device: {device}")

    adata = treedata.read_h5td(args.adata)
    adata.obs['time'] = [
        float(x.split('-')[0][1:]) for x in adata.obs['embryo'].values
    ]

    tp_indices, tp_tensors, node_info = precompute(adata, tree, device)
    print(f"  {len(node_info)} eligible nodes")

    node_mmd = compute_mmd(args, tree, tp_tensors, node_info, device)

    os.makedirs(args.outdir, exist_ok=True)
    outpath = os.path.join(args.outdir, f"node_mmd_{args.tree_index:03d}.pkl")
    with open(outpath, "wb") as f:
        pickle.dump(node_mmd, f)
    print(f"Done → {outpath}  ({len(node_mmd)} nodes)")


if __name__ == "__main__":
    main()