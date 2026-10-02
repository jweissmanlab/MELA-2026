# Setup
import argparse
import multiprocessing as mp
from pathlib import Path

import matplotlib.pyplot as plt
import pycea as py
import tracertools
import treedata as td
from tqdm import tqdm
from tracertools.config import edit_palette, set_theme

set_theme()

# Globals visible inside worker processes
_TDATA = None
_ARGS = None
_CLADE_TREES = None


def _init_clade_worker(tdata, args):
    global _TDATA, _ARGS
    _TDATA = tdata
    _ARGS = args


def _init_clone_worker(tdata, args, clade_trees):
    global _TDATA, _ARGS, _CLADE_TREES
    _TDATA = tdata
    _ARGS = args
    _CLADE_TREES = clade_trees


#### Helper function ####
def solve_clade(clade):
    """Reconstruct a clade subtree with FastTree using the clade's informative characters."""
    clade_characters = _TDATA[_TDATA.obs.clade == clade].obsm["characters"]
    use_characters = tracertools.utils.select_characters(clade_characters)
    tree = tracertools.solver.fasttree(clade_characters[use_characters], root_name=clade)
    return clade, (tree, use_characters)


def process_clone(clone):
    """Graft clade subtrees onto a clone stump and estimate branch lengths.

    Infers ancestral characters, collapses mutationless edges, estimates branch
    lengths with ConvexML, and counts per-branch edits.
    """
    clone_tree = _TDATA.obst[clone].copy()
    clone_tree = tracertools.tree.replace_subtrees(clone_tree, list(_CLADE_TREES.values()))
    tracertools.tree.ancestral_characters(clone_tree, _TDATA[_TDATA.obs.clone == clone].obsm["characters"])
    tracertools.tree.collapse_mutationless_edges(clone_tree, rescue_edges=True)
    tracertools.tree.estimate_branch_lengths(
        clone_tree, verbose=False, mutation_rates=[0.3, 0.4, 1.2], total_time=_ARGS.time, pseudo_count=1
    )
    tracertools.tree.count_branch_edits(clone_tree)
    return clone, clone_tree


def main():
    """Reconstruct clade and clone trees for one embryo, then save the tree plot and ``.h5td``."""
    parser = argparse.ArgumentParser(description="Embryo Tree Reconstruction")
    parser.add_argument("-i", "--input", help="Input file path", required=True)
    parser.add_argument("-o", "--output", help="Output file path", required=True)
    parser.add_argument("-n", "--name", help="Sample name", required=True)
    parser.add_argument("-t", "--time", help="Total time", type=float, required=True)

    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    prefix = args.name.lower().replace("-", "_")
    output_path = output_path / prefix
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    tdata = td.read_h5td(input_path / prefix / f"{prefix}_stump.h5td")

    # Reconstruct clades
    clades = tdata.obs["clade"].dropna().unique()
    clade_trees = {}
    clade_characters = {}

    with mp.Pool(processes=10, initializer=_init_clade_worker, initargs=(tdata, args)) as pool:
        for clade, (tree, use_characters) in tqdm(pool.imap(solve_clade, clades), total=len(clades)):
            clade_trees[clade] = tree
            clade_characters[clade] = use_characters

    tdata.uns["clade_characters"] = clade_characters

    # Process clones
    clones = tdata.obs["clone"].dropna().unique()
    with mp.Pool(processes=10, initializer=_init_clone_worker, initargs=(tdata, args, clade_trees)) as pool:
        results = tqdm(pool.imap(process_clone, clones), total=len(clones), desc="Processing clone trees")
        clone_trees = dict(results)

    for clone, clone_tree in clone_trees.items():
        tdata.obst[clone] = clone_tree

    # Plot
    fig, ax = plt.subplots(figsize=(5, 5), dpi=500)
    py.pl.tree(tdata, keys="characters", palette=edit_palette, ax=ax, depth_key="time")
    py.pl.annotation(tdata, keys="clade", width=0.3, legend=False, ax=ax)
    plt.savefig(output_path / f"{prefix}_tree.png", bbox_inches="tight")
    plt.close()

    # Save outputs
    tdata.write_h5td(output_path / f"{prefix}_tree.h5td")


if __name__ == "__main__":
    main()
