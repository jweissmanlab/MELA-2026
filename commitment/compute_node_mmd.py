import treedata
from geomloss import SamplesLoss
import numpy as np
import networkx as nx
from geomloss import SamplesLoss
from collections import defaultdict
from tqdm.notebook import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import scanpy as sc
import pickle

adata = treedata.read_h5td('/home/gokulg/orcd/scratch/lt/data/kl0.0_d50_l2_covnocov_adata_with_embeddings.h5ad')

tree_filenames = []
for e in ["7.5", "8.0", "8.5", "9.0", "9.5"]:
    for r in [1, 2, 3]:
        tree_filenames.append(f"e{e}_r{r}_tree")
tree_filenames.append("e10.0_r1_tree")

tdatas = [
    treedata.read_h5td(f"/home/gokulg/orcd/scratch/lt/data/embryos/{f}.h5td")
    for f in tree_filenames
]

trees, tree_names = [], []
for td in tdatas:
    for k in td.obst.keys():
        trees.append(td.obst[k])
        tree_names.append(k)



# Create name->index mapping
name_to_idx = {name: idx for idx, name in enumerate(adata.obs_names)}

# get state variables as numpy array for fast indexing
states = adata.obsm['X_scvi']

# MMD with energy kernel
mmd_loss = SamplesLoss("energy")

# get all leaves
leaves = []
for tree in trees:
    leaves.extend([node for node in tree.nodes() if tree.out_degree(node) == 0 and node in name_to_idx])

# precompute descendants for each node (cache to avoid recomputation)
descendants_cache = {}

def get_leaf_descendants(node, tree):
    """Get all leaf descendants of a node."""
    if node in descendants_cache:
        return descendants_cache[node]
    
    if tree.out_degree(node) == 0:  # node is a leaf
        result = {node}
    else:
        result = set()
        for descendant in nx.descendants(tree, node):
            if tree.out_degree(descendant) == 0:
                result.add(descendant)
    
    descendants_cache[node] = result
    return result

##################
# subsample for MMD reference distribution
##################
n_subsample = 100
##################
##################

# get all leaf indices and subsample once
all_leaf_indices = np.array([name_to_idx[leaf] for leaf in leaves])
all_leaf_states = states[all_leaf_indices]

# subsample all leaf states
if len(all_leaf_states) > n_subsample:
    subsample_idx = np.random.choice(len(all_leaf_states), n_subsample, replace=False)
    all_leaf_states_subsampled = all_leaf_states[subsample_idx]
else:
    all_leaf_states_subsampled = all_leaf_states

all_leaf_states_tensor = torch.from_numpy(all_leaf_states_subsampled).float()

# Compute MMD once per node
node_mmd = {}

for tree, name in tqdm(zip(trees, tree_names), desc="processing trees", total=len(trees)):

    nodes = tree.nodes()

    for node in tqdm(nodes, desc="computing MMD per node", leave=False):

        # get descendant leaves of this node
        descendant_leaves = get_leaf_descendants(node, tree)
        
        # get their indices and states
        descendant_indices = np.array([name_to_idx[l] for l in descendant_leaves if l in name_to_idx])

        if descendant_indices.size == 0:
            # print(f"Node {node} has no descendant leaves in the dataset, skipping MMD computation.")
            continue

        descendant_states = states[descendant_indices]
        
        # subsample descendant states
        if len(descendant_states) > n_subsample:
            subsample_idx = np.random.choice(len(descendant_states), n_subsample, replace=False)
            descendant_states_subsampled = descendant_states[subsample_idx]
        else:
            descendant_states_subsampled = descendant_states
        
        descendant_states_tensor = torch.from_numpy(descendant_states_subsampled).float()
        
        # compute MMD between entire population and descendant leaves
        mmd_value = mmd_loss(all_leaf_states_tensor, descendant_states_tensor).item()
        
        # get time attribute if it exists
        node_time = tree.nodes[node].get('time', None)
        
        node_mmd[node] = (mmd_value, node_time)

results = {}

for tree in tqdm(trees, desc="aggregating MMD results for leaves", total=len(trees)):
    leaves_in_tree = [node for node in tree.nodes() if tree.out_degree(node) == 0 and node in name_to_idx]
    for leaf in tqdm(leaves_in_tree, desc="processing leaves", leave=False):
        if leaf not in node_mmd:
            continue
        mmd_list = [node_mmd[leaf]]  # include leaf's own MMD
        
        # walk up the tree from leaf to root
        current = leaf
        
        # for each ancestor, look up the precomputed MMD
        while list(tree.predecessors(current)):
            parent = list(tree.predecessors(current))[0]
            if parent in node_mmd:
                mmd_list.append(node_mmd[parent])
            current = parent
        
        results[leaf] = mmd_list

# Define threshold as fraction of maximum MMD
threshold_fraction = 0.9

t_threshold_values = []

for leaf in tqdm(adata.obs_names, desc=f"Computing t_{threshold_fraction}"):

    if leaf not in results:
        # If no MMD results for this leaf, set to NaN
        t_threshold_values.append(np.nan)
        continue

    trajectory = results[leaf]
    
    if not trajectory:
        # If no trajectory, set to NaN
        t_threshold_values.append(np.nan)
        continue
    
    mmd_values, times = zip(*trajectory)
    mmd_values = np.array(mmd_values)
    times = np.array(times)
    
    # Find maximum MMD
    max_mmd = np.max(mmd_values)
    
    # Compute threshold
    threshold = threshold_fraction * max_mmd
    
    # Find first time when MMD exceeds threshold
    # Sort by time to ensure we find the earliest occurrence
    sort_idx = np.argsort(times)
    times_sorted = times[sort_idx]
    mmd_sorted = mmd_values[sort_idx]
    
    exceeds_threshold = mmd_sorted >= threshold
    
    if np.any(exceeds_threshold):
        # Find first index where threshold is exceeded
        first_idx = np.argmax(exceeds_threshold)
        t_threshold = times_sorted[first_idx]
    else:
        # Threshold never exceeded
        t_threshold = np.nan
    
    t_threshold_values.append(t_threshold)

# Add to adata.obs with dynamic column name
col_name = f'mmd_t{int(threshold_fraction*100)}'
adata.obs[col_name] = t_threshold_values

print(f"Added '{col_name}' to adata.obs")
print(f"Number of cells: {len(t_threshold_values)}")
print(f"Non-NaN values: {np.sum(~np.isnan(t_threshold_values))}")
print(f"Mean {col_name}: {np.nanmean(t_threshold_values):.4f}")
print(f"Median {col_name}: {np.nanmedian(t_threshold_values):.4f}")

# save the .obs as a csv
obs_df = adata.obs.copy()
obs_df.to_csv("/home/gokulg/orcd/scratch/devmap-paper/commitment/adata_obs_with_mmd_thresholds.csv")

# save the node to mmd dictionary with pickle
with open("/home/gokulg/orcd/scratch/devmap-paper/commitment/node_mmd_dictionary.pkl", "wb") as f:
    pickle.dump(node_mmd, f)