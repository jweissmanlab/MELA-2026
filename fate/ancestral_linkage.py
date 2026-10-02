#!/usr/bin/env python3

import argparse

import pycea as py

from devmap.config import get_paths
from devmap.utils import load_data

base_path, plots_path, results_path = get_paths("fate")


def main() -> None:
    """Compute permutation-tested ancestral linkage for one embryo and write the stats CSV."""
    parser = argparse.ArgumentParser(description="Run ancestral linkage analysis for a specified embryo.")
    parser.add_argument(
        "embryo",
        help='Embryo ID, for example: "E8.5-R1"',
    )
    parser.add_argument(
        "--groupby",
        default="cell_type",
        help="Groupby variable for linkage analysis (default: cell_type)",
    )
    args = parser.parse_args()

    embryo = args.embryo
    groupby = args.groupby

    tdata = load_data("topology")
    embryo_tdata = tdata[tdata.obs.embryo == embryo].copy()

    py.tl.ancestral_linkage(
        embryo_tdata,
        groupby,
        n_threads=32,
        depth_key="time",
        metric="lca",
        test="permutation",
        n_permutations=100,
        alternative="two-sided",
        permutation_mode="non_target",
    )
    stats = embryo_tdata.uns[f"{groupby}_symmetrized_linkage_stats"].copy()
    # make directory if it doesn't exist
    out_path = results_path / f"{groupby}_linkage"
    if not out_path.exists():
        out_path.mkdir(parents=True)

    stats.to_csv(out_path / f"{embryo}_linkage.csv")


if __name__ == "__main__":
    main()
