# Setup
import argparse
import multiprocessing as mp
from pathlib import Path

import pandas as pd
import pycea as py
import tracertools
import treedata as td
from tqdm import tqdm

from devmap.progenitors import identify_fate_progenitors

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


def _integration(character):
    """Integration id for a character, e.g. 'intID1039-EMX1' -> 'intID1039'."""
    return character.split("-", 1)[0]


#### Helper function ####
def solve_clade(clade):
    """Reconstruct a clade subtree with FastTree, leaving out the dropped integration's characters."""
    clade_characters = _TDATA[_TDATA.obs.clade == clade].obsm["characters"]
    use_characters = tracertools.utils.select_characters(clade_characters)
    # Jackknife: leave one integration (its three characters) out. Characters of
    # the dropped integration that are not in this clade's subset are simply
    # absent, so the clade keeps all of its remaining selected characters.
    dropped = _ARGS.dropped_integration
    clade_use = [c for c in use_characters if _integration(c) != dropped]
    # Guard against removing every informative character for a clade.
    if not clade_use:
        clade_use = list(use_characters)
    tree = tracertools.solver.fasttree(clade_characters[clade_use], root_name=clade)
    return clade, (tree, clade_use)


def process_clone(clone):
    """Graft clade subtrees onto a clone stump and estimate branch lengths.

    Node character annotations are removed afterwards to reduce output size.
    """
    clone_tree = _TDATA.obst[clone].copy()
    clone_tree = tracertools.tree.replace_subtrees(clone_tree, list(_CLADE_TREES.values()))
    tracertools.tree.ancestral_characters(clone_tree, _TDATA[_TDATA.obs.clone == clone].obsm["characters"])
    tracertools.tree.collapse_mutationless_edges(clone_tree, rescue_edges=True)
    tracertools.tree.estimate_branch_lengths(
        clone_tree, verbose=False, mutation_rates=[0.3, 0.4, 1.2], total_time=_ARGS.time, pseudo_count=1
    )
    tracertools.tree.count_branch_edits(clone_tree)
    for node in clone_tree.nodes:
        del clone_tree.nodes[node]["characters"]
    return clone, clone_tree


def progenitor_stats(tdata):
    """Summarize germ-layer fate progenitors across all clone trees.

    Returns
    -------
    pandas.DataFrame
        Per-fate mean progenitor time, mean fate-descendant count, and number of progenitors.
    """
    one_hot = pd.get_dummies(tdata.obs["germ_layer"]).astype(int)
    tdata.obsm["germ_layer_counts"] = one_hot
    tdata.obs["n"] = 1
    py.tl.ancestral_states(tdata, keys="germ_layer_counts", method="sum")
    py.tl.ancestral_states(tdata, keys="n", method="sum")
    clone_progenitors = {}
    for clone, tree in tdata.obst.items():
        print(f"Identifying progenitors for clone {clone}...")
        progenitors, _ = identify_fate_progenitors(tree, "germ_layer_counts", fate_names=one_hot.columns)
        clone_progenitors[clone] = progenitors.assign(clone=clone)
    progenitors = pd.concat(clone_progenitors.values())
    prog_stats = (
        progenitors.query("total_descendants > 1")
        .groupby(["fate"])
        .apply(
            lambda g: pd.Series(
                {
                    "mean_time": g["time"].mean(),
                    "mean_size": g["fate_descendants"].mean(),
                    "n_progenitors": g["node"].nunique(),
                }
            )
        )
        .reset_index()
    )
    return prog_stats


def main():
    """Run one leave-one-integration-out reconstruction and save topology, progenitor, and linkage results."""
    parser = argparse.ArgumentParser(description="Leave-one-integration-out jackknife of embryo tree reconstruction")
    parser.add_argument("-i", "--input", help="Input stump .h5td file path", required=True)
    parser.add_argument("-o", "--output", help="Output directory", required=True)
    parser.add_argument("-n", "--name", help="Sample name", required=True)
    parser.add_argument("-t", "--time", help="Total time", type=float, required=True)
    parser.add_argument(
        "-g", "--integration-index", help="Index of the integration to leave out", type=int, required=True
    )
    parser.add_argument("-p", "--processes", help="Worker processes", type=int, default=4)

    args = parser.parse_args()
    input_path = Path(args.input)
    prefix = args.name.lower().replace("-", "_")
    output_path = Path(args.output) / prefix / "jackknife"
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    tdata = td.read_h5td(input_path)

    # Identify the integration to leave out for this iteration (order-preserving
    # unique intIDs; each integration is a set of characters, e.g. 3 target sites).
    all_characters = list(tdata.obsm["characters"].columns)
    integrations = list(dict.fromkeys(_integration(c) for c in all_characters))
    if not (0 <= args.integration_index < len(integrations)):
        raise ValueError(
            f"integration_index {args.integration_index} out of range for {len(integrations)} integrations"
        )
    args.dropped_integration = integrations[args.integration_index]

    # Reconstruct clades (leaving out the dropped character)
    clades = tdata.obs["clade"].dropna().unique()
    clade_trees = {}
    clade_characters = {}

    with mp.Pool(processes=args.processes, initializer=_init_clade_worker, initargs=(tdata, args)) as pool:
        for clade, (tree, use_characters) in tqdm(pool.imap(solve_clade, clades), total=len(clades)):
            clade_trees[clade] = tree
            clade_characters[clade] = use_characters

    tdata.uns["clade_characters"] = clade_characters

    # Process clones
    clones = tdata.obs["clone"].dropna().unique()
    with mp.Pool(processes=args.processes, initializer=_init_clone_worker, initargs=(tdata, args, clade_trees)) as pool:
        results = tqdm(pool.imap(process_clone, clones), total=len(clones), desc="Processing clone trees")
        clone_trees = dict(results)

    for clone, clone_tree in clone_trees.items():
        tdata.obst[clone] = clone_tree

    # Get progenitor stats
    prog_stats = progenitor_stats(tdata)
    prog_stats = prog_stats.assign(
        integration_index=args.integration_index, dropped_integration=args.dropped_integration
    )
    linkage = py.tl.ancestral_linkage(tdata, "germ_layer", depth_key="time", copy=True)
    linkage = linkage.assign(integration_index=args.integration_index, dropped_integration=args.dropped_integration)

    # Save outputs
    del tdata.obsm["characters"]
    tdata.write_h5td(output_path / f"{prefix}_topology_{args.integration_index:02d}.h5td")
    prog_stats.to_csv(output_path / f"{prefix}_progenitor_stats_{args.integration_index:02d}.csv", index=False)
    linkage.to_csv(output_path / f"{prefix}_linkage_{args.integration_index:02d}.csv")


if __name__ == "__main__":
    main()
