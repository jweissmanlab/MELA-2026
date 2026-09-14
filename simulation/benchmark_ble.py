"""Benchmark branch-length estimation (BLE) accuracy.

On the *true* simulated topology, estimate node times with ConvexML
(tracertools.tree.estimate_branch_lengths) and measure the mean absolute error
(MAE) between estimated and true node times. Holds num_extant=10000 fixed and
sweeps:
  - division_sigma (lognormal sigma of the birth-waiting distribution)
        in {0.1, 0.2, 0.3, 0.4, 0.5}
  - number_of_cassettes in {5,10,15,20,25,30,35,40,45,50}
with 10 iterations each (5 x 10 x 10 = 500 independent Slurm array tasks).

The simulation is seeded by the iteration index. Per-job rows are written to
<outdir>/rows/ and the master CSV <outdir>/benchmark_ble_results.csv is rebuilt
(atomic rename) after each job, so it stays current as jobs complete without any
cross-job write races.
"""

import argparse
import glob
import os
import socket
import traceback

import numpy as np
import pandas as pd

import treedata as td  # noqa: F401  (registers accessors used downstream)
import pycea as py
import cassiopeia as cas
import tracertools

# --------------------------------------------------------------------------- #
# Benchmark grid
# --------------------------------------------------------------------------- #
NUM_EXTANT = 10000
CASSETTES = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
SIGMAS = [0.1, 0.2, 0.3, 0.4, 0.5]
N_ITERS = 10


def build_configs():
    """Return the full ordered list of (n_cassettes, division_sigma, iteration) jobs."""
    configs = []
    for n_cassettes in CASSETTES:
        for sigma in SIGMAS:
            for it in range(N_ITERS):
                configs.append((n_cassettes, sigma, it))
    return configs


# --------------------------------------------------------------------------- #
# Simulation helpers
# --------------------------------------------------------------------------- #
def get_mutation_rate(tdata, edit_frac):
    """Calculate mutation rate from edit fraction."""
    total_time = tdata.obs.time.values[0]
    return 1 - (1 - edit_frac) ** (1 / total_time)


def embryo_fitness(parent, rng=None):
    scale = 1
    if parent["time"] > 2:
        scale = 0.4
    return ({"birth_scale": scale}, {"birth_rate": scale})


def simulate(n_cassettes, sigma, seed):
    """Simulate a ground-truth tree with character matrix; rescale node times to [0, 1]."""
    tdata = cas.sim.birth_death_process(
        num_extant=NUM_EXTANT,
        on_division=embryo_fitness,
        random_seed=seed,
        birth_waiting_distribution=lambda scale, rng: rng.lognormal(
            mean=np.log(scale), sigma=sigma
        ),
    )
    cas.sim.stochastic_tracing(
        tdata,
        mutation_rate=get_mutation_rate(tdata, 0.5),
        number_of_cassettes=n_cassettes,
        random_seed=seed,
        state_priors={str(i): 1 / 8 for i in range(1, 9)},
    )
    cas.tl.rescale_node_times(tdata, max=1)
    return tdata


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def aggregate(outdir):
    """Rebuild the master CSV from all per-job row files (atomic rename)."""
    rows_dir = os.path.join(outdir, "rows")
    files = sorted(glob.glob(os.path.join(rows_dir, "*.csv")))
    if not files:
        return
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f))
        except Exception:
            # A row file may be mid-write by another job; skip it this round.
            continue
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    sort_cols = [c for c in ("n_cassettes", "division_sigma", "iteration") if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    master = os.path.join(outdir, "benchmark_ble_results.csv")
    tmp = f"{master}.tmp.{os.getpid()}"
    df.to_csv(tmp, index=False)
    os.replace(tmp, master)


# --------------------------------------------------------------------------- #
# Single job
# --------------------------------------------------------------------------- #
def run_job(n_cassettes, sigma, iteration, outdir):
    rows_dir = os.path.join(outdir, "rows")
    os.makedirs(rows_dir, exist_ok=True)

    record = {
        "n_cassettes": n_cassettes,
        "division_sigma": sigma,
        "iteration": iteration,
        "n_leaves": np.nan,
        "n_characters": np.nan,
        "mae": np.nan,
        "status": "ok",
        "error": "",
    }

    try:
        tdata = simulate(n_cassettes, sigma, seed=iteration)
        record["n_leaves"] = tdata.shape[0]
        record["n_characters"] = tdata.obsm["characters"].shape[1]

        tracertools.tree.estimate_branch_lengths(
            tdata.obst["simulated"], key_added="convexml_time", pseudo_count=1
        )
        node_df = py.get.node_df(tdata).query("time != 1").copy()
        record["mae"] = (abs(node_df["time"] - node_df["convexml_time"])).mean()
    except Exception as exc:
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    row_path = os.path.join(
        rows_dir, f"row_cas{n_cassettes}_sigma{sigma}_{iteration}.csv"
    )
    pd.DataFrame([record]).to_csv(row_path, index=False)
    print(
        f"[{record['status']}] cassettes={n_cassettes} sigma={sigma} "
        f"iter={iteration} mae={record['mae']}",
        flush=True,
    )

    aggregate(outdir)
    return record


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "results_ble"
        ),
        help="Directory for row files and the master CSV.",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Index into the (cassettes, sigma, iteration) grid. Defaults to "
        "$SLURM_ARRAY_TASK_ID when running under Slurm.",
    )
    parser.add_argument("--cassettes", type=int, help="Override: number_of_cassettes.")
    parser.add_argument("--sigma", type=float, help="Override: division_sigma.")
    parser.add_argument("--iter", type=int, help="Override: iteration index.")
    parser.add_argument(
        "--n-configs", action="store_true", help="Print grid size and exit."
    )
    args = parser.parse_args()

    configs = build_configs()
    if args.n_configs:
        print(len(configs))
        return

    if args.cassettes is not None and args.sigma is not None and args.iter is not None:
        n_cassettes, sigma, iteration = args.cassettes, args.sigma, args.iter
    else:
        task_id = args.task_id
        if task_id is None:
            env_id = os.environ.get("SLURM_ARRAY_TASK_ID")
            if env_id is None:
                parser.error(
                    "Provide --task-id (or --cassettes/--sigma/--iter), or run under "
                    "Slurm so $SLURM_ARRAY_TASK_ID is set."
                )
            task_id = int(env_id)
        if not 0 <= task_id < len(configs):
            parser.error(f"task-id {task_id} out of range [0, {len(configs)})")
        n_cassettes, sigma, iteration = configs[task_id]

    os.makedirs(args.outdir, exist_ok=True)
    run_job(n_cassettes, sigma, iteration, args.outdir)


if __name__ == "__main__":
    main()
