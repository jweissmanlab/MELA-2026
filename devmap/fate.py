from collections.abc import Sequence
from typing import Literal
import treedata as td
import numpy as np
import pandas as pd
import networkx as nx
import pycea as py
import anndata as ad
from scipy.sparse import issparse
from tqdm.auto import tqdm
import multiprocessing as mp
import sys
from collections.abc import Callable

def extant_node_attribute(
    tdata: td.TreeData,
    key: str | Sequence[str],
    depth_key: str = "depth",
    bins: int | Sequence[float] = 20,
    tree: str | Sequence[str] | None = None,
    column_names: Sequence[str] | None = None,
    extend_branches: bool = False,
    return_type: Literal["anndata", "dataframe"] = "anndata",
    paths: bool = False,
    sample: int | None = None,
    random_state: int | np.random.Generator | None = None,
) -> ad.AnnData | pd.DataFrame:
    """
    Expand one or more node attributes across depth bins.

    For every edge (parent → child), the child node is considered present
    at each depth bin that its incoming branch spans — from the parent's
    bin (inclusive) up to the child's bin (exclusive), unless
    ``extend_branches=True`` and the child is a leaf, in which case it
    extends to the last bin. The root is included at its own bin.

    When ``paths=True``, each root-to-leaf path is treated as a separate
    observation: every internal node appears once per path that passes
    through it, and each row is labelled with its leaf node ID.

    Parameters
    ----------
    tdata
        TreeData object.
    key
        Attribute(s) of node data to expand. When a single string is given
        the behaviour is unchanged. When a list is given, each attribute
        that stores a *scalar* per node becomes a single column named after
        the key; each attribute that stores a *list* per node expands into
        multiple columns named ``{key}_0``, ``{key}_1``, …
    depth_key
        Attribute storing depth values.
    bins
        Number of histogram bins or explicit bin edges.
    tree
        The ``obst`` key(s) of trees to use. If None, all trees are used.
    column_names
        Optional column name override(s). When ``key`` is a single string
        and that key holds a list, ``column_names`` must match the list
        length. When ``key`` is a list of strings, ``column_names`` must
        match the *total* number of expanded columns across all keys and
        is applied in order.
    extend_branches
        If True, leaf branches are extended to the maximum depth bin.
    return_type
        ``"anndata"`` (default) returns an :class:`~anndata.AnnData` where
        ``X`` contains all attribute columns, ``var_names`` are the column
        names, and ``obs`` holds ``depth_key``, ``node``, ``tree``,
        and ``leaf`` when ``paths=True``.
        ``"dataframe"`` returns the equivalent :class:`~pandas.DataFrame`.
    paths
        If True, emit one block of rows per root-to-leaf path rather than
        one block per node. Internal nodes appear once per path through
        them. A ``leaf`` column is added identifying the leaf at the end
        of each path.
    sample
        When ``paths=True``, sample this many paths total across all trees
        without first expanding unselected paths. Leaves are sampled
        proportionally across trees and only the selected paths are
        expanded. Ignored when ``paths=False``.
    random_state
        Seed or :class:`numpy.random.Generator` for reproducible sampling.
        Ignored when ``sample`` is None.

    Returns
    -------
    :class:`~anndata.AnnData` or :class:`~pandas.DataFrame` depending on
    ``return_type``.
    """
    import anndata as ad

    keys: list[str] = [key] if isinstance(key, str) else list(key)
    trees = py.utils.get_trees(tdata, tree)
    results: list[pd.DataFrame] = []
    col_names: list[str] | None = None

    rng = (
        np.random.default_rng(random_state)
        if not isinstance(random_state, np.random.Generator)
        else random_state
    )

    # ------------------------------------------------------------------
    # If sampling, draw leaves across trees before any row expansion.
    # Collect all leaves per tree, do a single proportional draw, then
    # only expand the selected paths.
    # ------------------------------------------------------------------
    sampled_leaves: dict[str, set] | None = None
    if paths and sample is not None:
        tree_leaves: dict[str, list] = {
            tk: [n for n in t.nodes if t.out_degree(n) == 0]
            for tk, t in trees.items()
        }
        total_leaves = sum(len(v) for v in tree_leaves.values())

        if sample >= total_leaves:
            sampled_leaves = {tk: leaves for tk, leaves in tree_leaves.items()}
        else:
            # Pool all (tree_key, leaf) pairs and sample directly
            all_pairs = [
                (tk, leaf)
                for tk, leaves in tree_leaves.items()
                for leaf in leaves
            ]
            chosen_idx = rng.choice(len(all_pairs), size=sample, replace=False)
            sampled_leaves = {tk: [] for tk in trees}
            for idx in chosen_idx:
                tk, leaf = all_pairs[idx]
                sampled_leaves[tk].append(leaf)

    def _make_row(node_id, bi, timepoints, nodes, tree_key, leaf_id=None):
        row: dict = {
            depth_key: timepoints[bi],
            "node": node_id,
            "tree": tree_key,
        }
        if paths:
            row["leaf"] = leaf_id
        col_iter = iter(col_names)
        for k in keys:
            val = nodes.loc[node_id, k]
            if np.isscalar(val):
                row[next(col_iter)] = val
            else:
                for v in val:
                    row[next(col_iter)] = v
        return row

    def _node_bin_range(u, v, t, nodes, timepoints):
        birth = nodes.loc[u, depth_key]
        death = nodes.loc[v, depth_key]
        birth_idx = np.searchsorted(timepoints, birth + 1e-6, side="left")
        is_leaf = t.out_degree(v) == 0
        if is_leaf and extend_branches:
            death_idx = len(timepoints)
        else:
            death_idx = np.searchsorted(timepoints, death + 1e-6, side="left")
        return birth_idx, death_idx

    # ------------------------------------------------------------------
    # Main loop over trees
    # ------------------------------------------------------------------
    for tree_key, t in trees.items():
        node_keys = [depth_key] + keys
        nodes = py.utils.get_keyed_node_data(tdata, keys=node_keys, tree=tree_key)
        nodes.index = nodes.index.droplevel("tree")

        # Resolve column names once from the first tree
        if col_names is None:
            expanded: list[str] = []
            for k in keys:
                first_val = nodes[k].iloc[0]
                if np.isscalar(first_val):
                    expanded.append(k)
                else:
                    n = len(first_val)
                    expanded.extend(f"{k}_{i}" for i in range(n))

            if column_names is not None:
                if len(column_names) != len(expanded):
                    raise ValueError(
                        f"column_names has length {len(column_names)} but the "
                        f"expanded key(s) produce {len(expanded)} columns."
                    )
                col_names = list(column_names)
            else:
                col_names = expanded

        timepoints = np.histogram_bin_edges(nodes[depth_key], bins=bins)
        root = py.utils.get_root(t)
        rows: list[dict] = []

        if not paths:
            for u, v in t.edges:
                birth_idx, death_idx = _node_bin_range(u, v, t, nodes, timepoints)
                for bi in range(birth_idx, death_idx):
                    rows.append(_make_row(v, bi, timepoints, nodes, tree_key))

            root_idx = np.searchsorted(
                timepoints, nodes.loc[root, depth_key], side="left"
            )
            rows.append(_make_row(root, root_idx, timepoints, nodes, tree_key))

        else:
            active_leaves = (
                sampled_leaves[tree_key]
                if sampled_leaves is not None
                else [n for n in t.nodes if t.out_degree(n) == 0]
            )

            for leaf in active_leaves:
                path_nodes = nx.shortest_path(t, root, leaf)

                root_idx = np.searchsorted(
                    timepoints, nodes.loc[root, depth_key], side="left"
                )
                rows.append(_make_row(root, root_idx, timepoints, nodes, tree_key, leaf_id=leaf))

                for u, v in zip(path_nodes[:-1], path_nodes[1:]):
                    birth_idx, death_idx = _node_bin_range(u, v, t, nodes, timepoints)
                    for bi in range(birth_idx, death_idx):
                        rows.append(_make_row(v, bi, timepoints, nodes, tree_key, leaf_id=leaf))

        if rows:
            results.append(pd.DataFrame(rows))

    df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    if return_type == "dataframe":
        return df

    obs_cols = [depth_key, "node", "tree"] + (["leaf"] if paths else [])
    obs = df[obs_cols].reset_index(drop=True)
    X = df[col_names].values
    var = pd.DataFrame(index=col_names)

    return ad.AnnData(X=X, obs=obs, var=var)


def commitment_index(
    adata: ad.AnnData,
    fate: str | None = None,
    depth_key: str = "time",
    tree_key: str = "tree",
    pseudocount: float = 1e-10,
    key_added: str = "commitment_index",
) -> None:
    """
    Compute commitment index as normalized entropy for each node.

    When *fate* is None, commitment index is defined as ``1 - H(p) / H(q)``
    where ``p`` is the node's descendant proportion vector and ``q`` is the
    root proportion vector for each tree.  A value of 0 indicates no
    commitment and 1 indicates complete commitment to a single fate.

    When *fate* is a var name, the count matrix is collapsed to a binary
    "fate k vs rest" vector and the same formula is applied using binary
    entropy, giving a per-fate commitment index.

    Parameters
    ----------
    adata
        AnnData object.  ``X`` contains descendant counts per fate,
        ``obs`` contains ``depth_key`` and ``tree_key``.
    fate
        A single ``var_name`` to compute one-vs-rest commitment for.
        If *None*, uses the full proportion vector (original behavior).
    depth_key
        Column in ``adata.obs`` storing depth / time values.
    tree_key
        Column in ``adata.obs`` identifying distinct trees.
    pseudocount
        Small value added to counts before normalization to avoid
        log(0).
    key_added
        Key added to ``adata.obs`` storing the resulting commitment indices.

    Returns
    -------
    None. Adds ``adata.obs[key_added]`` in place.
    """
    X_raw = adata.X

    if fate is not None:
        fate_idx = list(adata.var_names).index(fate)
        fate_counts = X_raw[:, fate_idx]
        rest_counts = X_raw.sum(axis=1) - fate_counts
        X = np.column_stack([fate_counts, rest_counts]) + pseudocount
    else:
        X = X_raw + pseudocount

    depths = adata.obs[depth_key].values
    trees = adata.obs[tree_key].values

    p = X / X.sum(axis=1, keepdims=True)
    H_p = -(p * np.log(p)).sum(axis=1)

    def _get_root_probs(X, depths, trees):
        """Compute root proportion vector per tree."""
        root_probs = {}
        for tree_name in np.unique(trees):
            mask = trees == tree_name
            tree_depths = depths[mask]
            root_rows = X[mask][tree_depths == tree_depths.min()]
            root_counts = root_rows.sum(axis=0)
            root_probs[tree_name] = root_counts / root_counts.sum()
        return root_probs

    root_probs = _get_root_probs(X, depths, trees)

    # Build per-row H_q via tree assignment
    unique_trees, tree_indices = np.unique(trees, return_inverse=True)
    H_q_per_tree = np.empty(len(unique_trees))
    for i, t in enumerate(unique_trees):
        q = root_probs[t]
        H_q_per_tree[i] = -(q * np.log(q)).sum()

    H_q = H_q_per_tree[tree_indices]
    bias = np.clip(1.0 - H_p / H_q, 0.0, 1.0)
    adata.obs[key_added] = bias

def permuted_commitment_control(
    adata: ad.AnnData,
    seed: int = 0,
    depth_key: str = "time",
    tree_key: str = "tree",
    n_key: str = "n",
) -> np.ndarray:
    """
    Generate a single permuted count matrix matching the structure of *adata*.

    Parameters
    ----------
    adata
        AnnData.  ``X`` holds fate counts, ``obs`` must contain
        *depth_key*, *tree_key*, and *n_key*.
    seed
        Random seed for reproducibility.
    depth_key
        Column in ``obs`` with depth / time values.
    tree_key
        Column in ``obs`` identifying distinct trees.
    n_key
        Column in ``obs`` with total descendant count per node.

    Returns
    -------
    ``(n_obs, n_vars)`` numpy array with multinomial-resampled fate counts.
    """
    rng = np.random.default_rng(seed)

    X_orig = adata.X
    depths = adata.obs[depth_key].values
    trees = adata.obs[tree_key].values
    n_desc = adata.obs[n_key].values.astype(int)

    def _get_root_probs(X, depths, trees):
        root_probs = {}
        for tree_name in np.unique(trees):
            mask = trees == tree_name
            tree_depths = depths[mask]
            root_rows = X[mask][tree_depths == tree_depths.min()]
            root_counts = root_rows.sum(axis=0)
            root_probs[tree_name] = root_counts / root_counts.sum()
        return root_probs

    root_probs = _get_root_probs(X_orig, depths, trees)

    unique_trees = np.unique(trees)
    tree_groups = {t: np.where(trees == t)[0] for t in unique_trees}

    tree_n_groups = {}
    for t in unique_trees:
        idxs = tree_groups[t]
        ns = n_desc[idxs]
        unique_ns, inv = np.unique(ns, return_inverse=True)
        tree_n_groups[t] = (idxs, unique_ns, inv, root_probs[t])

    X_perm = np.empty_like(X_orig)
    for t, (idxs, unique_ns, inv, p) in tree_n_groups.items():
        for j, n_val in enumerate(unique_ns):
            node_mask = inv == j
            count = node_mask.sum()
            samples = rng.multinomial(n_val, p, size=count)
            X_perm[idxs[node_mask]] = samples

    return X_perm


def permutation_test_commitment(
    adata: ad.AnnData,
    fate: str | list[str] | None = None,
    n_permutations: int = 100,
    seed: int = 0,
    group_cols: list[str] = ("time", "embryo", "stage"),
    depth_key: str = "time",
    tree_key: str = "tree",
    n_key: str = "n",
    pseudocount: float = 1e-10,
    root_probs: dict[str, np.ndarray] | np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Permutation test for commitment index, grouped by *group_cols*.

    Parameters
    ----------
    adata
        AnnData with fate counts in ``X`` and metadata in ``obs``.
    fate
        A single ``var_name``, a list of var names, or *None*.
        If *None*, uses the full proportion vector (original behavior).
        If a list, each fate is tested independently and permutations
        are reused across fates.
    n_permutations
        Number of permuted replicates.
    seed
        Random seed.
    group_cols
        Columns to group by when computing the weighted-mean
        commitment index.
    depth_key, tree_key, n_key, pseudocount
        Forwarded to helpers.
    root_probs
        Optional override for the root probability vector(s) used as
        the H_q baseline in the commitment index.

        - ``None`` (default): root probs are inferred from the
          minimum-depth rows of each tree, as before.
        - ``np.ndarray`` of shape ``(n_fates,)`` or ``(n_fates_binary,)``
          (i.e. a single probability vector): the same vector is used
          for every tree.
        - ``dict[str, np.ndarray]``: a mapping from tree name to its
          probability vector.  Every tree present in
          ``adata.obs[tree_key]`` must have an entry.

        When *fate* is a string (binary collapse), the vector should
        have length 2 (``[p_fate, p_other]``); when *fate* is *None*
        it should have length ``adata.n_vars``.

    Returns
    -------
    summary : DataFrame
        One row per group (× fate) with columns: *group_cols*,
        (``fate``), ``commitment_index`` (observed), ``mean_permuted``,
        ``std_permuted``, ``p_value``.
    permutations : DataFrame
        One row per group (× fate) × permutation.
    """
    group_cols = list(group_cols)

    if fate is None or isinstance(fate, str):
        fate_list = [fate]
    else:
        fate_list = list(fate)
    multi_fate = not (len(fate_list) == 1 and fate_list[0] is None)

    n_desc = adata.obs[n_key].values
    obs = adata.obs
    group_keys = [obs[c] for c in group_cols]

    denom = pd.Series(n_desc.astype(float), index=obs.index).groupby(group_keys).sum()

    def _get_root_probs(X, depths, trees):
        """Infer root probs from minimum-depth rows (original behavior)."""
        result = {}
        for tree_name in np.unique(trees):
            mask = trees == tree_name
            tree_depths = depths[mask]
            root_rows = X[mask][tree_depths == tree_depths.min()]
            root_counts = root_rows.sum(axis=0)
            result[tree_name] = root_counts / root_counts.sum()
        return result

    def _resolve_root_probs(X, depths, trees, fate_name):
        """
        Return a root-prob dict for the current X layout, respecting
        the ``root_probs`` parameter when provided.
        """
        if root_probs is None:
            return _get_root_probs(X, depths, trees)

        unique_trees = np.unique(trees)

        if isinstance(root_probs, np.ndarray):
            # Broadcast a single vector to every tree.
            vec = np.asarray(root_probs, dtype=float)
            vec = vec / vec.sum()          # normalise defensively
            return {t: vec for t in unique_trees}

        # dict path — validate keys and normalise.
        missing = set(unique_trees) - set(root_probs.keys())
        if missing:
            raise ValueError(
                f"root_probs is missing entries for tree(s): {missing}"
            )
        return {
            t: np.asarray(root_probs[t], dtype=float) / np.asarray(root_probs[t]).sum()
            for t in unique_trees
        }

    def _ci_from_X(X_raw, fate_name):
        if fate_name is not None:
            fate_idx = list(adata.var_names).index(fate_name)
            fc = X_raw[:, fate_idx]
            rc = X_raw.sum(axis=1) - fc
            X = np.column_stack([fc, rc]) + pseudocount
        else:
            X = X_raw + pseudocount

        depths = obs[depth_key].values
        trees = obs[tree_key].values
        resolved = _resolve_root_probs(X, depths, trees, fate_name)

        p = X / X.sum(axis=1, keepdims=True)
        H_p = -(p * np.log(p)).sum(axis=1)

        unique_trees, tree_indices = np.unique(trees, return_inverse=True)
        H_q_arr = np.empty(len(unique_trees))
        for k, t in enumerate(unique_trees):
            q = resolved[t]
            H_q_arr[k] = -(q * np.log(q)).sum()

        return np.clip(1.0 - H_p / H_q_arr[tree_indices], 0.0, 1.0)

    def _weighted_mean_ci(ci_values):
        return (
            pd.Series(ci_values * n_desc.astype(float), index=obs.index)
            .groupby(group_keys)
            .sum()
            / denom
        )

    # --- Observed ---
    wm_obs_by_fate = {
        fn: _weighted_mean_ci(_ci_from_X(adata.X, fn))
        for fn in fate_list
    }

    # --- Permutations ---
    n_groups = len(denom)
    perm_wms = {f: np.empty((n_permutations, n_groups)) for f in fate_list}

    for i in tqdm(range(n_permutations), desc="Permutations"):
        pX = permuted_commitment_control(
            adata, seed=seed + i,
            depth_key=depth_key, tree_key=tree_key, n_key=n_key,
        )
        for fn in fate_list:
            perm_wms[fn][i] = _weighted_mean_ci(_ci_from_X(pX, fn)).values

    # --- Assemble output DataFrames ---
    summary_parts = []
    perm_parts = []

    for fate_name in fate_list:
        wm_obs = wm_obs_by_fate[fate_name]
        obs_vals = wm_obs.values
        pw = perm_wms[fate_name]

        mean_perm = pw.mean(axis=0)
        std_perm = pw.std(axis=0, ddof=1)
        p_values = (pw >= obs_vals[None, :]).mean(axis=0)

        df_summary = wm_obs.reset_index(name="commitment_index")
        df_summary["mean_permuted"] = mean_perm
        df_summary["std_permuted"] = std_perm
        df_summary["p_value"] = p_values

        idx = wm_obs.reset_index().drop(columns=0, errors="ignore")
        if "commitment_index" in idx.columns:
            idx = idx.drop(columns="commitment_index")
        perm_rows = []
        for i in range(n_permutations):
            row = idx.copy()
            row["commitment_index"] = pw[i]
            row["permutation"] = i
            perm_rows.append(row)
        df_perms = pd.concat(perm_rows, ignore_index=True)

        if multi_fate:
            label = fate_name if fate_name is not None else "all"
            df_summary["fate"] = label
            df_perms["fate"] = label

        summary_parts.append(df_summary)
        perm_parts.append(df_perms)

    summary = pd.concat(summary_parts, ignore_index=True)
    permutations = pd.concat(perm_parts, ignore_index=True)

    return summary, permutations