# Setup
import argparse
from pathlib import Path

import cassiopeia as cas
import networkx as nx
import numpy as np
import pandas as pd
import pycea as py

from devmap.progenitors import identify_fate_progenitors


def permute_labels(labels, fraction, rng):
    """Randomly permute `germ_layer` labels among a random `fraction` of leaves.

    Models leaf misplacement: a fraction of leaves have their fate labels
    shuffled amongst each other, mimicking cells placed at the wrong tip.
    """
    labels = np.asarray(labels).copy()
    n = len(labels)
    k = int(round(fraction * n))
    if k > 1:
        sel = rng.choice(n, size=k, replace=False)
        labels[sel] = labels[sel][rng.permutation(k)]
    return labels


def embryo_fitness(parent, rng=None):
    """Division callback that slows birth rate (scale 0.4) after time 2."""
    scale = 1
    if parent["time"] > 2:
        scale = 0.4
    return ({"birth_scale": scale}, {"birth_rate": scale})


def germ_layer_trajectory():
    """Fate trajectory used to assign germ layers along the simulated tree."""
    traj = nx.DiGraph()
    traj.add_edges_from(
        [
            ("root", "uncommitted"),
            ("uncommitted", "endoderm"),
            ("uncommitted", "mesectoderm"),
            ("mesectoderm", "ectoderm"),
            ("mesectoderm", "mesoderm"),
        ]
    )
    nx.set_node_attributes(
        traj,
        {"root": 0, "uncommitted": 0.6, "endoderm": 1, "mesectoderm": 0.7, "ectoderm": 1, "mesoderm": 1},
        "time",
    )
    nx.set_node_attributes(
        traj,
        {"root": [0], "uncommitted": [0], "endoderm": [1], "mesectoderm": [0], "ectoderm": [2], "mesoderm": [3]},
        "X_latent",
    )
    nx.set_edge_attributes(
        traj,
        {
            ("root", "uncommitted"): 1,
            ("uncommitted", "endoderm"): 0.2,
            ("uncommitted", "mesectoderm"): 0.8,
            ("mesectoderm", "ectoderm"): 0.6,
            ("mesectoderm", "mesoderm"): 0.4,
        },
        "prob",
    )
    return traj


def simulate_tree(seed, num_extant):
    """Simulate a ground-truth tree with germ-layer fates.

    Deterministic in `seed`, so the same topology and fates are produced
    regardless of misplacement rate.
    """
    tdata = cas.sim.birth_death_process(
        num_extant=num_extant,
        on_division=embryo_fitness,
        random_seed=seed,
        birth_waiting_distribution=lambda scale, rng: rng.lognormal(mean=np.log(scale), sigma=0.1),
    )
    cas.tl.rescale_node_times(tdata, max=1)
    tdata = cas.sim.trajectory_expression(tdata, germ_layer_trajectory(), n_genes=1, random_seed=seed, prob_key="prob")
    tdata.obs["germ_layer"] = tdata.obsm["X_latent"]["0"].map({0: pd.NA, 1: "endoderm", 2: "ectoderm", 3: "mesoderm"})
    return tdata


def realized_misplacement(tdata, key="germ_layer"):
    """Compute the realized (effective) misplacement rate.

    This is the fraction of leaves whose (permuted) label disagrees with the
    Fitch-Hartigan reconstructed state of their parent.
    """
    tree = tdata.obst["simulated"]
    py.tl.ancestral_states(tdata, keys=key, method="fitch_hartigan", keys_added="fitch_germ_layer")
    misplaced = 0
    for leaf in tdata.obs_names:
        parent = next(iter(tree.predecessors(leaf)))
        if tdata.obs.loc[leaf, key] != tree.nodes[parent]["fitch_germ_layer"]:
            misplaced += 1
    return misplaced / tdata.n_obs


def progenitor_stats(tdata):
    """Summarize germ-layer fate progenitors across all trees.

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
    """Simulate one topology and compute progenitor and linkage stats across misplacement rates."""
    parser = argparse.ArgumentParser(
        description="Simulated leaf-misplacement stability test for progenitor / linkage metrics"
    )
    parser.add_argument("-o", "--output", help="Output directory", required=True)
    parser.add_argument("-s", "--seed", help="Topology index (random seed for simulation)", type=int, required=True)
    parser.add_argument(
        "-r", "--rates", help="Misplacement rates in percent", type=float, nargs="+", default=[0, 1, 2, 5, 10, 20]
    )
    parser.add_argument("--num-extant", help="Number of leaves to simulate", type=int, default=10000)

    args = parser.parse_args()
    output_path = Path(args.output) / "simulation"
    output_path.mkdir(parents=True, exist_ok=True)

    # Simulate the ground-truth tree once; identical across all misplacement rates.
    tdata = simulate_tree(args.seed, args.num_extant)
    truth = np.asarray(tdata.obs["germ_layer"])

    prog_results = []
    link_results = []
    for pct in args.rates:
        rng = np.random.default_rng([int(args.seed), int(pct)])
        tdata.obs["germ_layer"] = permute_labels(truth, pct / 100.0, rng)
        misplaced = realized_misplacement(tdata)

        prog = progenitor_stats(tdata).assign(topology=args.seed, rate=pct, realized_misplacement=misplaced)
        prog_results.append(prog)

        link = py.tl.ancestral_linkage(tdata, "germ_layer", depth_key="time", copy=True)
        link = link.reset_index(names="germ_layer").assign(
            topology=args.seed, rate=pct, realized_misplacement=misplaced
        )
        link_results.append(link)

    topo_tag = f"t{int(args.seed):02d}"
    pd.concat(prog_results).to_csv(output_path / f"simulation_progenitor_stats_{topo_tag}.csv", index=False)
    pd.concat(link_results).to_csv(output_path / f"simulation_linkage_{topo_tag}.csv", index=False)


if __name__ == "__main__":
    main()
