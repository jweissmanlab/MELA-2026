import networkx as nx
import pandas as pd
import numpy as np

def identify_fate_progenitors(
    tree: nx.DiGraph,
    key: str = "germ_layer_counts",
    fate_names: list[str] | None = None,
    ignored_fates: list[str] | None = None,
    time_attr: str = "time",
    threshold: float = 0.9,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify progenitor nodes for each fate category in a lineage tree,
    correctly handling polytomies to avoid overcounting due to unresolved
    cell divisions.

    Supports two modes, auto-detected from the first node's value for *key*:

      **Multi-fate mode** — value is a list/array of ints (counts per fate).
        *fate_names* is required and must match the length of the counts
        vector.  Each fate is processed independently.

      **Single-fate mode** — value is a scalar float (fraction for one fate).
        *fate_names* is ignored; the single fate is named after *key*.
        The fraction is used directly (no denominator needed).

    A node can be a progenitor for multiple fates (e.g. at a polytomy where
    one child is ecto-committed and another is meso-committed, the parent
    is the progenitor for both).  However, a progenitor's descendants are
    pruned *per-fate*: if a node is selected as a progenitor for fate F,
    no descendant can also be a progenitor for F — but descendants CAN
    still be progenitors for other fates.

    Algorithm
    ---------
    For each fate independently:

      Pass 1 — bottom-up qualification:
        A node qualifies for fate F if EITHER:
          (a) It is a leaf whose fate-F fraction >= threshold, OR
          (b) It is an internal node whose fate-F fraction >= threshold
              (direct), OR
          (c) It is a polytomy (>2 children) with >1 children that
              qualified *directly* for fate F (bubble-up, single level).

      Pass 2 — top-down selection (per-fate):
        Traverse in topological order. Accept the highest qualifying node
        for fate F; prune its subtree from further consideration *for
        fate F only*.

    Finally, results across all fates are combined into a single DataFrame.

    Parameters
    ----------
    tree : nx.DiGraph
        Rooted lineage tree.  Each node must have:
          - *key*: either a list/array of ints (multi-fate) or a float
            (single-fate).
          - *time_attr*: a scalar time annotation.
    key : str
        Node attribute name for the fate data.  Its type on the first node
        determines the operating mode.
    fate_names : list[str] | None
        Names for each fate category (multi-fate mode only).  Length must
        match the counts vectors.  Ignored in single-fate mode.
    ignored_fates : list[str] | None
        Names of fates (must be a subset of *fate_names*) that should be
        excluded from progenitor identification — no progenitors will ever
        be assigned for these fates.  However, their counts ARE still
        included in the total-descendants denominator used to compute
        *fate_fraction*, so purity scores for the active fates remain
        correct.  Only valid in multi-fate mode; raises ValueError if
        provided in single-fate mode.
    time_attr : str
        Node attribute name for time.  Defaults to "time".
    threshold : float
        Fraction of descendants required for a node to qualify directly.
        Defaults to 0.9.

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        DataFrame — one row per (progenitor, fate) pair with columns:
          node              - node identifier
          fate              - fate category name
          fate_descendants  - # leaves of this fate claimed by this progenitor
          total_descendants - # total leaves claimed by this progenitor
          fate_fraction     - fate_descendants / total_descendants
          time              - node's time attribute
          n_children        - number of direct children
          qualified_via     - 'direct' or 'bubble_up'

        DataFrame — index is leaf node with columns:
          progenitor  - assigned progenitor node (or None)
          fate        - assigned fate name (or None)
    """
    if not nx.is_directed_acyclic_graph(tree):
        raise ValueError("tree must be a DAG.")

    roots = [n for n in tree.nodes if tree.in_degree(n) == 0]
    if len(roots) != 1:
        raise ValueError(f"Expected a single root, found {len(roots)}.")

    topo_order = list(nx.topological_sort(tree))

    # -- Auto-detect mode from the first node's value ----------------------
    first_node = topo_order[0]
    sample_val = tree.nodes[first_node][key]

    if isinstance(sample_val, (list, tuple, np.ndarray)):
        # Multi-fate mode (counts vector)
        multi_fate = True
        if fate_names is None:
            raise ValueError(
                "fate_names is required in multi-fate mode "
                "(node attribute is a list/array)."
            )
        n_fates = len(fate_names)
        if len(sample_val) != n_fates:
            raise ValueError(
                f"Length of counts vector ({len(sample_val)}) on node "
                f"{first_node!r} does not match fate_names ({n_fates})."
            )

        # Validate and resolve ignored_fates to index set
        ignored_fates = ignored_fates or []
        unknown = set(ignored_fates) - set(fate_names)
        if unknown:
            raise ValueError(
                f"ignored_fates contains names not found in fate_names: {unknown}"
            )
        ignored_indices = {fate_names.index(f) for f in ignored_fates}

    elif isinstance(sample_val, (int, float, np.integer, np.floating)):
        # Single-fate mode (scalar fraction)
        if ignored_fates:
            raise ValueError(
                "ignored_fates is not supported in single-fate mode."
            )
        multi_fate = False
        fate_names = [key]
        n_fates = 1
        ignored_indices = set()
    else:
        raise TypeError(
            f"Unsupported type for key {key!r} on node {first_node!r}: "
            f"{type(sample_val).__name__}.  Expected list/array or scalar."
        )

    # -- helpers -----------------------------------------------------------
    if multi_fate:
        def _counts(n):
            return tree.nodes[n][key]

        def _total(n):
            return sum(_counts(n))

        def _fate_frac(n, fi):
            t = _total(n)
            return _counts(n)[fi] / t if t > 0 else 0.0
    else:
        def _fate_frac(n, fi):
            return float(tree.nodes[n][key])

    # -- Pass 1: bottom-up qualification per fate --------------------------
    qualifies = [{} for _ in range(n_fates)]
    via = [{} for _ in range(n_fates)]
    bubble_sources = [{} for _ in range(n_fates)]

    for fi in range(n_fates):
        if fi in ignored_indices:
            # Mark every node as non-qualifying; skip all processing.
            for node in topo_order:
                qualifies[fi][node] = False
                via[fi][node] = None
            continue

        for node in reversed(topo_order):
            children = list(tree.successors(node))
            frac = _fate_frac(node, fi)
            is_leaf = len(children) == 0

            # (a) leaf
            if is_leaf:
                if frac >= threshold:
                    qualifies[fi][node] = True
                    via[fi][node] = "direct"
                else:
                    qualifies[fi][node] = False
                    via[fi][node] = None
                continue

            # (b) internal node passes threshold directly
            if frac >= threshold:
                qualifies[fi][node] = True
                via[fi][node] = "direct"
                continue

            # (c) bubble-up through polytomies only, single level
            if len(children) > 2:
                direct_qual = [
                    c for c in children
                    if qualifies[fi][c] and via[fi][c] == "direct"
                ]
                if len(direct_qual) > 1:
                    qualifies[fi][node] = True
                    via[fi][node] = "bubble_up"
                    bubble_sources[fi][node] = direct_qual
                    continue

            qualifies[fi][node] = False
            via[fi][node] = None

    # -- Helpers for leaf collection ---------------------------------------
    def _leaves_below(n):
        """All leaf descendants of n (including n if it is a leaf)."""
        desc = nx.descendants(tree, n) | {n}
        return {d for d in desc if tree.out_degree(d) == 0}

    # -- Pass 2: top-down selection with per-fate pruning ------------------
    all_records = []
    skip_fate = [set() for _ in range(n_fates)]
    progenitor_leaves = [{} for _ in range(n_fates)]

    for node in topo_order:
        node_fates = [
            fi for fi in range(n_fates)
            if fi not in ignored_indices
            and qualifies[fi][node]
            and node not in skip_fate[fi]
        ]
        if not node_fates:
            continue

        for fi in node_fates:
            if via[fi][node] == "direct":
                claimed = _leaves_below(node)
                pruned = nx.descendants(tree, node)
            else:
                claimed = set()
                pruned = set()
                for c in bubble_sources[fi][node]:
                    claimed.update(_leaves_below(c))
                    pruned.add(c)
                    pruned.update(nx.descendants(tree, c))

            skip_fate[fi].update(pruned)
            progenitor_leaves[fi][node] = claimed

            fate_desc = sum(
                1 for lf in claimed if _fate_frac(lf, fi) >= threshold
            )
            total_desc = len(claimed)

            all_records.append({
                "node":              node,
                "fate":              fate_names[fi],
                "fate_descendants":  fate_desc,
                "total_descendants": total_desc,
                "fate_fraction":     (
                    round(fate_desc / total_desc, 4) if total_desc > 0 else 0.0
                ),
                "time":              tree.nodes[node][time_attr],
                "n_children":        tree.out_degree(node),
                "qualified_via":     via[fi][node],
            })

    # -- Build leaf -> progenitor mapping ----------------------------------
    all_leaves = [n for n in tree.nodes if tree.out_degree(n) == 0]

    leaf_records = {lf: {"progenitor": None, "fate": None} for lf in all_leaves}
    for fi in range(n_fates):
        for prog, leaves in progenitor_leaves[fi].items():
            for lf in leaves:
                leaf_records[lf] = {"progenitor": prog, "fate": fate_names[fi]}

    leaf_df = pd.DataFrame.from_dict(leaf_records, orient="index")
    leaf_df.index.name = "leaf"

    df = pd.DataFrame(all_records, columns=[
        "node", "fate", "fate_descendants", "total_descendants",
        "fate_fraction", "time", "n_children", "qualified_via",
    ])

    return df, leaf_df

def compute_pmi(df, group_col, cat_col, min_count=1, smoothing=0.0):
    """
    Compute PMI and NPMI for category co-occurrence across groups.

    Parameters
    ----------
    df : pd.DataFrame
    group_col : str
    cat_col : str
    min_count : int
        Minimum co-occurrence count to keep (helps reduce noise)
    smoothing : float
        Additive smoothing to avoid log(0); e.g., 1e-9

    Returns
    -------
    cooc : DataFrame (counts)
    pmi : DataFrame
    npmi : DataFrame
    """

    # 1. Remove duplicates (important!)
    df = df.drop_duplicates([group_col, cat_col])

    # 2. Build group × category matrix (binary)
    X = pd.crosstab(df[group_col], df[cat_col])

    # 3. Co-occurrence counts
    cooc = X.T @ X  # shape: (n_cat, n_cat)

    # 4. Total number of groups
    n_groups = X.shape[0]

    # 5. Convert to probabilities
    # P(i, j): probability two categories co-occur in a group
    P_ij = cooc / n_groups

    # P(i): marginal probability a category appears in a group
    P_i = np.diag(P_ij)

    # Add smoothing to avoid division/log issues
    if smoothing > 0:
        P_ij = P_ij + smoothing
        P_i = P_i + smoothing

    # Outer product for independence assumption
    P_i_P_j = np.outer(P_i, P_i)

    # 6. PMI
    with np.errstate(divide='ignore', invalid='ignore'):
        pmi = np.log(P_ij / P_i_P_j)

    pmi = pd.DataFrame(pmi, index=cooc.index, columns=cooc.columns)

    # 7. NPMI (normalized PMI)
    with np.errstate(divide='ignore', invalid='ignore'):
        npmi = pmi / (-np.log(P_ij))

    npmi = pd.DataFrame(npmi, index=cooc.index, columns=cooc.columns)

    # 8. Optional: filter low-count pairs
    if min_count > 1:
        mask = cooc >= min_count
        pmi = pmi.where(mask)
        npmi = npmi.where(mask)

    return cooc, pmi, npmi