#!/usr/bin/env python3

import argparse
import pandas as pd
import treedata as td
import pycea as py
from pathlib import Path
from devmap.config import get_paths, load_data

base_path, plots_path, results_path = get_paths("fate")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ancestral linkage analysis for a specified embryo."
    )
    parser.add_argument(
        "embryo",
        help='Embryo ID, for example: "E8.5-R1"',
    )
    parser.add_argument(
        "--groupby",
        default="cell_type",
        help='Groupby variable for linkage analysis (default: cell_type)',
    )
    args = parser.parse_args()

    embryo = args.embryo
    groupby = args.groupby

    tdata = load_data("topology")
    embryo_tdata = tdata[tdata.obs.embryo == embryo].copy()

    stats = py.tl.ancestral_linkage(
        embryo_tdata,
        groupby,
        n_threads=32,
        copy=True,
        depth_key="time",
        metric="path",
        test="permutation",
        n_permutations=100,
        alternative="two-sided",
        permutation_mode="non_target"
    )
    # make directory if it doesn't exist
    out_path = results_path / f"{groupby}_linkage"
    if not out_path.exists(): 
        out_path.mkdir(parents=True)

    stats.to_csv(
        out_path / f"{embryo}_linkage.csv"
    )


if __name__ == "__main__":
    main()