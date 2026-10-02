from collections import Counter
from collections.abc import Sequence
from typing import Literal

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import pycea as py
import treedata as td
from tqdm.auto import tqdm


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

    rng = np.random.default_rng(random_state) if not isinstance(random_state, np.random.Generator) else random_state

    # ------------------------------------------------------------------
    # If sampling, draw leaves across trees before any row expansion.
    # Collect all leaves per tree, do a single proportional draw, then
    # only expand the selected paths.
    # ------------------------------------------------------------------
    sampled_leaves: dict[str, set] | None = None
    if paths and sample is not None:
        tree_leaves: dict[str, list] = {tk: [n for n in t.nodes if t.out_degree(n) == 0] for tk, t in trees.items()}
        total_leaves = sum(len(v) for v in tree_leaves.values())

        if sample >= total_leaves:
            sampled_leaves = dict(tree_leaves.items())
        else:
            # Pool all (tree_key, leaf) pairs and sample directly
            all_pairs = [(tk, leaf) for tk, leaves in tree_leaves.items() for leaf in leaves]
            chosen_idx = rng.choice(len(all_pairs), size=sample, replace=False)
            sampled_leaves = {tk: [] for tk in trees}
            for idx in chosen_idx:
                tk, leaf = all_pairs[idx]
                sampled_leaves[tk].append(leaf)

    def _make_row(node_id, bi, timepoints, nodes, tree_key, leaf_id=None):
        """Build one output row for ``node_id`` at depth bin ``bi``, expanding list-valued keys into columns."""
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
        """Return the ``(start, stop)`` bin indices spanned by the branch from parent ``u`` to child ``v``."""
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

            root_idx = np.searchsorted(timepoints, nodes.loc[root, depth_key], side="left")
            rows.append(_make_row(root, root_idx, timepoints, nodes, tree_key))

        else:
            active_leaves = (
                sampled_leaves[tree_key] if sampled_leaves is not None else [n for n in t.nodes if t.out_degree(n) == 0]
            )

            for leaf in active_leaves:
                path_nodes = nx.shortest_path(t, root, leaf)

                root_idx = np.searchsorted(timepoints, nodes.loc[root, depth_key], side="left")
                rows.append(_make_row(root, root_idx, timepoints, nodes, tree_key, leaf_id=leaf))

                for u, v in zip(path_nodes[:-1], path_nodes[1:], strict=False):
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


def _advance_extant_branch(
    t: nx.DiGraph,
    node,
    target_bin: int,
    node_bins: dict,
    extend_branches: bool,
) -> list:
    """Return branches descended from ``node`` that are extant at target_bin."""
    children = list(t.successors(node))

    # The incoming branch remains extant until the node's depth bin.
    if node_bins[node] > target_bin:
        return [node]

    if not children:
        return [node] if extend_branches else []

    extant = []
    stack = children.copy()

    while stack:
        descendant = stack.pop()
        descendant_children = list(t.successors(descendant))

        if node_bins[descendant] > target_bin:
            extant.append(descendant)
        elif descendant_children:
            stack.extend(descendant_children)
        elif extend_branches:
            extant.append(descendant)

    return extant


def extant_transition_count(
    tdata: td.TreeData,
    key: str,
    depth_key: str = "depth",
    bins: int | Sequence[float] = 20,
    tree: str | Sequence[str] | None = None,
    extend_branches: bool = False,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Count categorical transitions between extant branches at consecutive depth bins.

    Each branch extant at the target timepoint contributes exactly one
    transition from its ancestral branch at the preceding timepoint.
    Consequently, branching between two timepoints can produce multiple
    transitions from one source branch.

    Parameters
    ----------
    tdata
        TreeData object.
    key
        Categorical node attribute carried by each node's incoming branch.
    depth_key
        Node attribute storing depth values.
    bins
        Number of histogram bins or explicit bin edges.
    tree
        Tree key or keys to use. If None, all trees are used.
    extend_branches
        If True, terminal branches remain extant through the final bin.
        Otherwise, they disappear when their terminal depth is reached.
    dropna
        If True, transitions involving missing categories are omitted.

    Returns
    -------
    pandas.DataFrame
        Columns are ``source_time``, ``target_time``, ``source``,
        ``target``, and ``count``.
    """
    trees = py.utils.get_trees(tdata, tree)
    transitions: Counter = Counter()

    for tree_key, t in trees.items():
        nodes = py.utils.get_keyed_node_data(
            tdata,
            keys=[depth_key, key],
            tree=tree_key,
        )
        nodes.index = nodes.index.droplevel("tree")

        timepoints = np.histogram_bin_edges(nodes[depth_key], bins=bins)
        root = py.utils.get_root(t)

        # Match extant_node_attribute's bin assignment.
        node_bins = {
            node: np.searchsorted(
                timepoints,
                nodes.loc[node, depth_key] + 1e-6,
                side="left",
            )
            for node in t.nodes
        }

        root_bin = np.searchsorted(
            timepoints,
            nodes.loc[root, depth_key],
            side="left",
        )

        # Active values are the nodes whose incoming branches are extant.
        # The root is treated as a special branch at its initial bin.
        active = [root]

        for source_idx in range(root_bin, len(timepoints) - 1):
            target_idx = source_idx + 1
            next_active = []

            for source_node in active:
                descendants = _advance_extant_branch(
                    t=t,
                    node=source_node,
                    target_bin=target_idx,
                    node_bins=node_bins,
                    extend_branches=extend_branches,
                )

                source_category = nodes.loc[source_node, key]

                for target_node in descendants:
                    target_category = nodes.loc[target_node, key]

                    if dropna and (pd.isna(source_category) or pd.isna(target_category)):
                        continue

                    transitions[
                        (
                            timepoints[source_idx],
                            timepoints[target_idx],
                            source_category,
                            target_category,
                        )
                    ] += 1

                next_active.extend(descendants)

            active = next_active

            if not active:
                break

    columns = [
        "source_time",
        "target_time",
        "source",
        "target",
        "count",
    ]

    return pd.DataFrame(
        [(*transition, count) for transition, count in transitions.items()],
        columns=columns,
    )


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
        """Compute the root fate proportion vector per tree from its minimum-depth rows."""
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
    for _t, (idxs, unique_ns, inv, p) in tree_n_groups.items():
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
        """Return a root-prob dict for the current X layout, respecting ``root_probs`` when provided."""
        if root_probs is None:
            return _get_root_probs(X, depths, trees)

        unique_trees = np.unique(trees)

        if isinstance(root_probs, np.ndarray):
            # Broadcast a single vector to every tree.
            vec = np.asarray(root_probs, dtype=float)
            vec = vec / vec.sum()  # normalise defensively
            return {t: vec for t in unique_trees}

        # dict path — validate keys and normalise.
        missing = set(unique_trees) - set(root_probs.keys())
        if missing:
            raise ValueError(f"root_probs is missing entries for tree(s): {missing}")
        return {t: np.asarray(root_probs[t], dtype=float) / np.asarray(root_probs[t]).sum() for t in unique_trees}

    def _ci_from_X(X_raw, fate_name):
        """Compute the per-row commitment index from a count matrix, optionally collapsed to one-vs-rest."""
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
        """Average commitment indices within each group, weighted by descendant count ``n_key``."""
        return pd.Series(ci_values * n_desc.astype(float), index=obs.index).groupby(group_keys).sum() / denom

    # --- Observed ---
    wm_obs_by_fate = {fn: _weighted_mean_ci(_ci_from_X(adata.X, fn)) for fn in fate_list}

    # --- Permutations ---
    n_groups = len(denom)
    perm_wms = {f: np.empty((n_permutations, n_groups)) for f in fate_list}

    for i in tqdm(range(n_permutations), desc="Permutations"):
        pX = permuted_commitment_control(
            adata,
            seed=seed + i,
            depth_key=depth_key,
            tree_key=tree_key,
            n_key=n_key,
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


def annotate_commitment(tree, leaf_categories, threshold=0.9, exclude_categories=()):
    """
    Annotate nodes with the most specific category exceeding ``threshold``.

    Leaf categories are aggregated bottom-up so that each node holds, for
    every annotation level, the category counts of its descendant leaves.
    A node is labelled with the category from the most specific (highest
    index) level whose descendant fraction is strictly greater than
    ``threshold``; if no level qualifies the node is ``"Uncommitted"``
    (level ``-1``). A top-down pass then propagates labels so that a child
    whose commitment level is not more specific than its parent's
    (``child_level <= parent_level``) inherits the parent's label and level.

    Missing values do not contribute to either the numerator or denominator.
    Excluded categories contribute to the denominator but cannot be selected
    as a commitment label.

    Parameters
    ----------
    tree : nx.DiGraph
        Lineage tree. Modified in place.
    leaf_categories : sequence of dict
        One mapping per annotation level, ordered from least to most
        specific (e.g. germ layer, lineage, cell type). Each maps leaf node
        IDs to a category; leaves absent from a mapping or with missing
        values are ignored at that level.
    threshold : float
        Minimum fraction of descendant leaves (exclusive) that a category
        must reach for a node to be committed to it.
    exclude_categories : iterable
        Categories that count toward the denominator but are never
        assigned as a commitment label.

    Returns
    -------
    None. Sets the node attributes ``"commitment"`` (category label or
    ``"Uncommitted"``) and ``"commitment_level"`` (index into
    ``leaf_categories``, or ``-1``) on every node of ``tree``.
    """
    exclude_categories = set(exclude_categories)
    n_levels = len(leaf_categories)

    level_counts = {}
    level_totals = {}

    def leaf_labels(node):
        """Return per-level category counters and totals (0 or 1) for a single leaf."""
        counts = []
        totals = []

        for category_map in leaf_categories:
            label = category_map.get(node, pd.NA)

            if pd.isna(label):
                counts.append(Counter())
                totals.append(0)
            else:
                counts.append(Counter([label]))
                totals.append(1)

        return counts, totals

    def most_specific_commitment(counts_per_level, totals_per_level):
        """Return ``(level, label)`` for the most specific level with a non-excluded category above threshold."""
        for level in range(n_levels - 1, -1, -1):
            total = totals_per_level[level]

            if total == 0:
                continue

            label = next(
                (
                    label
                    for label, count in counts_per_level[level].most_common()
                    if label not in exclude_categories and count / total > threshold
                ),
                None,
            )

            if label is not None:
                return level, label

        return -1, "Uncommitted"

    # Bottom-up pass
    for node in reversed(list(nx.topological_sort(tree))):
        children = list(tree.successors(node))

        if not children:
            level_counts[node], level_totals[node] = leaf_labels(node)
        else:
            counts_per_level = [Counter() for _ in range(n_levels)]
            totals_per_level = [0] * n_levels

            for child in children:
                for level in range(n_levels):
                    counts_per_level[level].update(level_counts[child][level])
                    totals_per_level[level] += level_totals[child][level]

            level_counts[node] = counts_per_level
            level_totals[node] = totals_per_level

        level, label = most_specific_commitment(
            level_counts[node],
            level_totals[node],
        )
        tree.nodes[node]["commitment"] = label
        tree.nodes[node]["commitment_level"] = level

    # Top-down pass
    for node in nx.topological_sort(tree):
        parent_label = tree.nodes[node]["commitment"]
        parent_level = tree.nodes[node]["commitment_level"]

        for child in tree.successors(node):
            child_level = tree.nodes[child]["commitment_level"]

            if child_level <= parent_level:
                tree.nodes[child]["commitment"] = parent_label
                tree.nodes[child]["commitment_level"] = parent_level


def get_commitment_path(tdata, key="commitment", key_added="commitment_path"):
    """
    Record the sequence of commitment labels along each leaf's ancestry.

    For every leaf of every tree in ``tdata.obst``, walks from the leaf up
    to the root and collects the distinct values of the node attribute
    ``"commitment"`` (as set by :func:`annotate_commitment`), in
    leaf-to-root order keeping the first occurrence of each label.

    Parameters
    ----------
    tdata : td.TreeData
        TreeData object whose trees carry a ``"commitment"`` node attribute.
    key : str
        Currently unused; the node attribute ``"commitment"`` is always read.
    key_added : str
        Column in ``tdata.obs`` in which to store the result.

    Returns
    -------
    None. Adds ``tdata.obs[key_added]`` in place, containing a
    comma-separated string of commitment labels from most to least specific
    (leaf to root). Cells not found as leaves in any tree get NaN.
    """
    combined_result = {}
    for tree in tdata.obst.values():
        leaves = [n for n in tree.nodes if tree.out_degree(n) == 0]
        result = {}

        for leaf in leaves:
            commitments = []
            current = leaf

            while True:
                commitment = tree.nodes[current].get("commitment")
                if commitment is not None and commitment not in commitments:
                    commitments.append(commitment)

                parents = list(tree.predecessors(current))
                if not parents:
                    break  # reached root

                current = parents[0]  # tree assumption: one parent per node

            result[leaf] = ", ".join(map(str, commitments))
        combined_result.update(result)
    tdata.obs[key_added] = tdata.obs_names.map(combined_result)
