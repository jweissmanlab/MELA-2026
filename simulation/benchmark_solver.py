"""Benchmark lineage tree reconstruction solvers.

Each invocation runs a SINGLE (num_extant, solver, iteration) combination so that
every job can be dispatched as an independent Slurm array task. The simulation is
seeded by the iteration index, so all solvers operating on the same
(num_extant, iteration) pair reconstruct from an identical ground-truth tree.

For each run we record the wall-clock time and the peak resident memory used by
the solver. Memory is sampled across the full process tree so that solvers which
shell out to subprocesses (e.g. FastTree) are accounted for.

Results are written one row per job to ``<outdir>/rows/`` and the master CSV
``<outdir>/benchmark_results.csv`` is rebuilt (atomic rename) after each job, so
it stays current as jobs complete without any cross-job write races.
"""

import argparse
import glob
import os
import socket
import threading
import time
import traceback

import numpy as np
import networkx as nx
import pandas as pd

import treedata as td  # noqa: F401  (registers accessors used downstream)
import pycea as py
import cassiopeia as cas
import tracertools
import pylaml

# --------------------------------------------------------------------------- #
# Benchmark grid
# --------------------------------------------------------------------------- #
SIZES = [200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
SOLVERS = ["nj", "upgma", "greedy", "fasttree", "hybrid", "laml"]
N_ITERS = 10


def build_configs():
    """Return the full ordered list of (size, solver, iteration) jobs."""
    configs = []
    for size in SIZES:
        for solver in SOLVERS:
            for it in range(N_ITERS):
                configs.append((size, solver, it))
    return configs


# --------------------------------------------------------------------------- #
# Simulation helpers (ported from simulate.py)
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


def simulate(num_extant, seed):
    """Simulate a ground-truth tree with character matrix and missing data."""
    tdata = cas.sim.birth_death_process(
        num_extant=num_extant,
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
        number_of_cassettes=35,
        random_seed=seed,
        state_priors={str(i): 1 / 8 for i in range(1, 9)},
    )
    #cas.sim.missing_data(tdata, stochastic_rate=0.05, random_seed=seed)
    return tdata


# --------------------------------------------------------------------------- #
# Solvers (ported from simulate.py). Each stores its tree under obst[name].
# --------------------------------------------------------------------------- #
def fasttree(tdata):
    tdata.obst["fasttree"] = tracertools.solver.fasttree(tdata.obsm["characters"])


def hybrid(tdata):
    tdata.obst["stump"] = tracertools.solver.n_mutation_greedy(tdata.obsm["characters"])[0]
    tdata.obs["clade"] = py.get.node_df(tdata, tree="stump")["parent"]
    clade_trees = {}
    for clade in tdata.obs["clade"].dropna().unique():
        clade_characters = tdata[tdata.obs["clade"] == clade].obsm["characters"]
        use_characters = tracertools.utils.select_characters(clade_characters)
        clade_trees[clade] = tracertools.solver.fasttree(
            clade_characters[use_characters], root_name=clade
        )
    tdata.obst["hybrid"] = tracertools.tree.replace_subtrees(
        tdata.obst["stump"], list(clade_trees.values()), error_on_missing=True
    )

def laml(tdata):
    characters = tdata.obsm["characters"]
    initial_tree = tracertools.solver.fasttree(characters)
    # Convert node names to integers for the LAML solver
    node_to_int = {}
    i = 0
    for node in tdata.obs_names:
        node_to_int[node] = i
        i += 1
    for node in initial_tree.nodes:
        if node not in node_to_int:
            node_to_int[node] = i
            i += 1
    int_to_node = {v: k for k, v in node_to_int.items()}
    # Format input for the LAML solver
    edges = list(nx.dfs_edges(initial_tree,source = "root"))
    tree = pylaml.make_tree(
        edges=[(node_to_int[edge[0]], node_to_int[edge[1]]) for edge in edges]
        ,branch_lengths=[1] * len(edges) + [0.0],num_leaves = tdata.shape[0])
    char_matrix = characters.astype(str).replace(
        "*", "0").replace("-", "-1").astype(np.int32).values
    priors = np.ones((char_matrix.shape[1], 8)) * .125
    # Perform the LAML topology search
    result = pylaml.topology_search(
        tree=tree,
        character_matrix=char_matrix,
        max_iterations=5000,
        initial_phi=0.01,
        ultrametric=True,
        mutation_priors=priors,
        verbose=False
    )
    tdata.obst["laml"] = nx.DiGraph([(int_to_node[edge[0]], int_to_node[edge[1]]) 
                                     for edge in result.optimized_tree['edges']])

SOLVER_FUNCS = {
    "nj": lambda x: cas.solver.nj(x),
    "upgma": lambda x: cas.solver.upgma(x),
    "greedy": lambda x: cas.solver.greedy(x),
    "fasttree": fasttree,
    "hybrid": hybrid,
    "laml": laml
}


# --------------------------------------------------------------------------- #
# Memory-tracked execution
# --------------------------------------------------------------------------- #
def run_with_memory(func, tdata, interval=0.05):
    """Run ``func(tdata)`` while sampling peak RSS of the full process tree.

    Returns (elapsed_seconds, baseline_rss, peak_rss) in bytes.
    """
    import psutil

    proc = psutil.Process()

    def tree_rss():
        total = proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                total += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total

    baseline = tree_rss()
    peak = baseline
    stop = threading.Event()

    def sampler():
        nonlocal peak
        while not stop.is_set():
            try:
                rss = tree_rss()
                if rss > peak:
                    peak = rss
            except Exception:
                pass
            stop.wait(interval)

    sampler_thread = threading.Thread(target=sampler, daemon=True)
    start = time.perf_counter()
    sampler_thread.start()
    try:
        func(tdata)
    finally:
        elapsed = time.perf_counter() - start
        stop.set()
        sampler_thread.join()
    return elapsed, baseline, peak


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
    df = df.sort_values(["n_leaves", "solver", "iteration"]).reset_index(drop=True)
    master = os.path.join(outdir, "benchmark_results.csv")
    tmp = f"{master}.tmp.{os.getpid()}"
    df.to_csv(tmp, index=False)
    os.replace(tmp, master)


# --------------------------------------------------------------------------- #
# Single job
# --------------------------------------------------------------------------- #
def run_job(num_extant, solver, iteration, outdir):
    rows_dir = os.path.join(outdir, "rows")
    os.makedirs(rows_dir, exist_ok=True)

    record = {
        "solver": solver,
        "iteration": iteration,
        "n_leaves": np.nan,
        "rf_norm": np.nan,
        "time_sec": np.nan,
        "solver_mem_mb": np.nan,
        "status": "ok",
        "error": "",
    }

    try:
        tdata = simulate(num_extant, seed=iteration)
        record["n_leaves"] = tdata.shape[0]

        elapsed, baseline, peak = run_with_memory(SOLVER_FUNCS[solver], tdata)
        record["time_sec"] = elapsed
        record["solver_mem_mb"] = peak / 1e6

        rf, rf_max = cas.critique.robinson_foulds(tdata, key1="simulated", key2=solver)
        record["rf_norm"] = rf / rf_max if rf_max else np.nan
    except Exception as exc:
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    row_path = os.path.join(rows_dir, f"row_{num_extant}_{solver}_{iteration}.csv")
    pd.DataFrame([record]).to_csv(row_path, index=False)
    print(
        f"[{record['status']}] size={num_extant} solver={solver} iter={iteration} "
        f"time={record['time_sec']:.2f}s mem={record['solver_mem_mb']:.1f}MB "
        f"rf_norm={record['rf_norm']}",
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
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
        help="Directory for row files and the master CSV.",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Index into the (size, solver, iteration) grid. Defaults to "
        "$SLURM_ARRAY_TASK_ID when running under Slurm.",
    )
    parser.add_argument("--size", type=int, help="Override: num_extant.")
    parser.add_argument("--solver", help="Override: solver name.")
    parser.add_argument("--iter", type=int, help="Override: iteration index.")
    parser.add_argument(
        "--n-configs",
        action="store_true",
        help="Print the total number of grid configurations and exit.",
    )
    args = parser.parse_args()

    configs = build_configs()
    if args.n_configs:
        print(len(configs))
        return

    if args.size is not None and args.solver is not None and args.iter is not None:
        size, solver, iteration = args.size, args.solver, args.iter
    else:
        task_id = args.task_id
        if task_id is None:
            env_id = os.environ.get("SLURM_ARRAY_TASK_ID")
            if env_id is None:
                parser.error(
                    "Provide --task-id (or --size/--solver/--iter), or run under "
                    "Slurm so $SLURM_ARRAY_TASK_ID is set."
                )
            task_id = int(env_id)
        if not 0 <= task_id < len(configs):
            parser.error(f"task-id {task_id} out of range [0, {len(configs)})")
        size, solver, iteration = configs[task_id]

    os.makedirs(args.outdir, exist_ok=True)
    run_job(size, solver, iteration, args.outdir)


if __name__ == "__main__":
    main()
