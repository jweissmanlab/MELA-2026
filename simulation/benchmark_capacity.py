"""Benchmark FastTree reconstruction accuracy vs. recording capacity.

Holds num_extant=10000 and solver=fasttree fixed and sweeps:
  - number_of_cassettes in {10,15,20,25,30,35,40,45,50}
  - missing-data stochastic_rate in {0, 0.05, 0.1, 0.15}
with 10 iterations each (9 x 4 x 10 = 360 independent Slurm array tasks).

The simulation is seeded by the iteration index, so the underlying birth-death
tree is identical across (cassettes, rate) for a given iteration; only the
character matrix (cassette count) and dropout (missing rate) vary.

Only reconstruction accuracy (Robinson-Foulds) is recorded -- not time or
memory. Per-job rows are written to <outdir>/rows/ and the master CSV
<outdir>/benchmark_capacity_results.csv is rebuilt (atomic rename) after each
job, so it stays current as jobs complete without any cross-job write races.
"""

import argparse
import glob
import os
import socket
import traceback

import numpy as np
import pandas as pd

import treedata as td  # noqa: F401  (registers accessors used downstream)
import cassiopeia as cas
import tracertools

# --------------------------------------------------------------------------- #
# Benchmark grid
# --------------------------------------------------------------------------- #
NUM_EXTANT = 10000
SOLVER = "fasttree"
CASSETTES = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
MISSING_RATES = [0.0, 0.05, 0.1, 0.15, .20]
N_ITERS = 10


def build_configs():
    """Return the full ordered list of (n_cassettes, missing_rate, iteration) jobs."""
    configs = []
    for n_cassettes in CASSETTES:
        for rate in MISSING_RATES:
            for it in range(N_ITERS):
                configs.append((n_cassettes, rate, it))
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


def simulate(n_cassettes, missing_rate, seed):
    """Simulate a ground-truth tree, character matrix, and (optional) dropout."""
    tdata = cas.sim.birth_death_process(
        num_extant=NUM_EXTANT,
        on_division=embryo_fitness,
        random_seed=seed,
        birth_waiting_distribution=lambda scale, rng: rng.lognormal(
            mean=np.log(scale), sigma=0.1
        ),
    )
    rate = get_mutation_rate(tdata, 0.5)
    cas.sim.stochastic_tracing(
        tdata,
        mutation_rate=rate,
        number_of_cassettes=n_cassettes,
        random_seed=seed,
        state_priors={str(i): 1 / 8 for i in range(1, 9)},
    )
    if missing_rate > 0:
        cas.sim.missing_data(tdata, stochastic_rate=missing_rate, random_seed=seed)
    return tdata


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #
def fasttree(tdata):
    tdata.obst["fasttree"] = tracertools.solver.fasttree(tdata.obsm["characters"])


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
    sort_cols = [c for c in ("n_cassettes", "missing_rate", "iteration") if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    master = os.path.join(outdir, "benchmark_capacity_results.csv")
    tmp = f"{master}.tmp.{os.getpid()}"
    df.to_csv(tmp, index=False)
    os.replace(tmp, master)


# --------------------------------------------------------------------------- #
# Single job
# --------------------------------------------------------------------------- #
def run_job(n_cassettes, missing_rate, iteration, outdir):
    rows_dir = os.path.join(outdir, "rows")
    os.makedirs(rows_dir, exist_ok=True)

    record = {
        "solver": SOLVER,
        "n_cassettes": n_cassettes,
        "missing_rate": missing_rate,
        "iteration": iteration,
        "n_leaves": np.nan,
        "n_characters": np.nan,
        "rf_norm": np.nan,
        "status": "ok",
        "error": "",
    }

    try:
        tdata = simulate(n_cassettes, missing_rate, seed=iteration)
        record["n_leaves"] = tdata.shape[0]
        record["n_characters"] = tdata.obsm["characters"].shape[1]

        fasttree(tdata)

        rf, rf_max = cas.critique.robinson_foulds(tdata, key1="simulated", key2=SOLVER)
        record["rf_norm"] = rf / rf_max if rf_max else np.nan
    except Exception as exc:
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    row_path = os.path.join(
        rows_dir, f"row_cas{n_cassettes}_miss{missing_rate}_{iteration}.csv"
    )
    pd.DataFrame([record]).to_csv(row_path, index=False)
    print(
        f"[{record['status']}] cassettes={n_cassettes} missing={missing_rate} "
        f"iter={iteration} rf_norm={record['rf_norm']}",
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
            os.path.dirname(os.path.abspath(__file__)), "results_capacity"
        ),
        help="Directory for row files and the master CSV.",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Index into the (cassettes, missing_rate, iteration) grid. Defaults "
        "to $SLURM_ARRAY_TASK_ID when running under Slurm.",
    )
    parser.add_argument("--cassettes", type=int, help="Override: number_of_cassettes.")
    parser.add_argument("--missing-rate", type=float, help="Override: stochastic_rate.")
    parser.add_argument("--iter", type=int, help="Override: iteration index.")
    parser.add_argument(
        "--n-configs", action="store_true", help="Print grid size and exit."
    )
    args = parser.parse_args()

    configs = build_configs()
    if args.n_configs:
        print(len(configs))
        return

    if (
        args.cassettes is not None
        and args.missing_rate is not None
        and args.iter is not None
    ):
        n_cassettes, missing_rate, iteration = (
            args.cassettes,
            args.missing_rate,
            args.iter,
        )
    else:
        task_id = args.task_id
        if task_id is None:
            env_id = os.environ.get("SLURM_ARRAY_TASK_ID")
            if env_id is None:
                parser.error(
                    "Provide --task-id (or --cassettes/--missing-rate/--iter), or run "
                    "under Slurm so $SLURM_ARRAY_TASK_ID is set."
                )
            task_id = int(env_id)
        if not 0 <= task_id < len(configs):
            parser.error(f"task-id {task_id} out of range [0, {len(configs)})")
        n_cassettes, missing_rate, iteration = configs[task_id]

    os.makedirs(args.outdir, exist_ok=True)
    run_job(n_cassettes, missing_rate, iteration, args.outdir)


if __name__ == "__main__":
    main()
