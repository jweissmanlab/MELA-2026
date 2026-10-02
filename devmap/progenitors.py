import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pycea as py
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from statsmodels.stats.multitest import multipletests


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
    """  # noqa: D205
    if not nx.is_directed_acyclic_graph(tree):
        raise ValueError("tree must be a DAG.")

    roots = [n for n in tree.nodes if tree.in_degree(n) == 0]
    if len(roots) != 1:
        raise ValueError(f"Expected a single root, found {len(roots)}.")

    topo_order = list(nx.topological_sort(tree))

    # -- Auto-detect mode from the first node's value ----------------------
    first_node = topo_order[0]
    sample_val = tree.nodes[first_node][key]

    if isinstance(sample_val, list | tuple | np.ndarray):
        # Multi-fate mode (counts vector)
        multi_fate = True
        if fate_names is None:
            raise ValueError("fate_names is required in multi-fate mode " "(node attribute is a list/array).")
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
            raise ValueError(f"ignored_fates contains names not found in fate_names: {unknown}")
        ignored_indices = {fate_names.index(f) for f in ignored_fates}

    elif isinstance(sample_val, int | float | np.integer | np.floating):
        # Single-fate mode (scalar fraction)
        if ignored_fates:
            raise ValueError("ignored_fates is not supported in single-fate mode.")
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
            """Per-fate counts vector stored on node n."""
            return tree.nodes[n][key]

        def _total(n):
            """Total count across all fates (including ignored fates) for node n."""
            return sum(_counts(n))

        def _fate_frac(n, fi):
            """Fraction of node n's counts belonging to fate index fi (0.0 if total is 0)."""
            t = _total(n)
            return _counts(n)[fi] / t if t > 0 else 0.0
    else:

        def _fate_frac(n, fi):
            """Scalar fate fraction stored on node n (fi is ignored in single-fate mode)."""
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
                direct_qual = [c for c in children if qualifies[fi][c] and via[fi][c] == "direct"]
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
            fi
            for fi in range(n_fates)
            if fi not in ignored_indices and qualifies[fi][node] and node not in skip_fate[fi]
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

            fate_desc = sum(1 for lf in claimed if _fate_frac(lf, fi) >= threshold)
            total_desc = len(claimed)

            all_records.append(
                {
                    "node": node,
                    "fate": fate_names[fi],
                    "fate_descendants": fate_desc,
                    "total_descendants": total_desc,
                    "fate_fraction": (round(fate_desc / total_desc, 4) if total_desc > 0 else 0.0),
                    "time": tree.nodes[node][time_attr],
                    "n_children": tree.out_degree(node),
                    "qualified_via": via[fi][node],
                }
            )

    # -- Build leaf -> progenitor mapping ----------------------------------
    all_leaves = [n for n in tree.nodes if tree.out_degree(n) == 0]

    leaf_records = {lf: {"progenitor": None, "fate": None} for lf in all_leaves}
    for fi in range(n_fates):
        for prog, leaves in progenitor_leaves[fi].items():
            for lf in leaves:
                leaf_records[lf] = {"progenitor": prog, "fate": fate_names[fi]}

    leaf_df = pd.DataFrame.from_dict(leaf_records, orient="index")
    leaf_df.index.name = "leaf"

    df = pd.DataFrame(
        all_records,
        columns=[
            "node",
            "fate",
            "fate_descendants",
            "total_descendants",
            "fate_fraction",
            "time",
            "n_children",
            "qualified_via",
        ],
    )

    return df, leaf_df


def get_fate_progenitors(tdata, key, key_added="progenitors", min_descendants=2):
    """
    Identify fate-restricted progenitor nodes across all clone trees in a TreeData.

    Runs :func:`identify_fate_progenitors` (with its default threshold of 0.9)
    on every tree in ``tdata.obst``, concatenates the per-clone results, and
    annotates each progenitor with its clone, stage, and embryo. Leaves are
    assigned to the progenitor that claims them and the assignment is written
    to ``tdata.obs[key_added]``.

    If *key* is in ``tdata.obsm`` it is treated as a per-fate counts table:
    its columns are used as the fate names (multi-fate mode). In that case, if
    ``tdata.obs["n"]`` does not exist, it is created (all 1) and summed onto
    the tree nodes with ``py.tl.ancestral_states``. Otherwise *key* is treated
    as a scalar fate-fraction node attribute (single-fate mode). In both cases
    the tree nodes must already carry a *key* attribute and a ``time``
    attribute.

    Parameters
    ----------
    tdata : td.TreeData
        TreeData with per-clone trees in ``.obst``. Modified in place.
    key : str
        Node attribute holding fate counts (if also a key in ``tdata.obsm``)
        or a scalar fate fraction.
    key_added : str
        Column in ``tdata.obs`` where each leaf's assigned progenitor node is
        stored. Defaults to "progenitors".
    min_descendants : int | None
        Keep only progenitors with ``fate_descendants >= min_descendants``;
        leaves assigned to discarded progenitors are left unassigned. If None,
        no filtering is applied. Defaults to 2.

    Returns
    -------
    pd.DataFrame
        One row per (progenitor, fate) pair, indexed by node, with the columns
        returned by :func:`identify_fate_progenitors` plus:
          clone   - clone ID (key in ``tdata.obst``)
          stage   - portion of the clone ID before "-R" (e.g. "E8.5")
          embryo  - portion of the clone ID before "-C" (e.g. "E8.5-R1")
    """
    is_counts = key in tdata.obsm
    if is_counts:
        fate_names = tdata.obsm[key].columns
        if "n" not in tdata.obs:
            tdata.obs["n"] = 1
            py.tl.ancestral_states(tdata, keys="n", method="sum")
    else:
        fate_names = None

    clone_progenitors = {}
    leaf_assignments = []
    for clone, tree in tdata.obst.items():
        print(f"Processing clone {clone}...")
        progenitors, clone_assignments = identify_fate_progenitors(
            tree,
            key,
            fate_names=fate_names,
        )
        clone_progenitors[clone] = progenitors.assign(clone=clone)
        leaf_assignments.append(clone_assignments)

    progenitors = pd.concat(clone_progenitors.values())
    leaf_assignments = pd.concat(leaf_assignments)
    progenitors["stage"] = progenitors["clone"].str.split("-R").str[0]
    progenitors["embryo"] = progenitors["clone"].str.split("-C").str[0]

    if min_descendants is not None:
        progenitors = progenitors.query("fate_descendants >= @min_descendants").copy()
        leaf_assignments = leaf_assignments.query("progenitor in @progenitors.node").copy()

    tdata.obs[key_added] = leaf_assignments["progenitor"]
    progenitors.index = progenitors["node"].values

    return progenitors


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
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(P_ij / P_i_P_j)

    pmi = pd.DataFrame(pmi, index=cooc.index, columns=cooc.columns)

    # 7. NPMI (normalized PMI)
    with np.errstate(divide="ignore", invalid="ignore"):
        npmi = pmi / (-np.log(P_ij))

    npmi = pd.DataFrame(npmi, index=cooc.index, columns=cooc.columns)

    # 8. Optional: filter low-count pairs
    if min_count > 1:
        mask = cooc >= min_count
        pmi = pmi.where(mask)
        npmi = npmi.where(mask)

    return cooc, pmi, npmi


def leaves_below(G, node):
    """
    Return the leaf descendants of a node.

    Parameters
    ----------
    G : nx.DiGraph
        Rooted tree.
    node : hashable
        Node whose descendant leaves are returned.

    Returns
    -------
    list
        Nodes in ``nx.descendants(G, node)`` with out-degree 0. The query node
        itself is not included, even if it is a leaf.
    """
    descendants = nx.descendants(G, node)
    leaves = [n for n in descendants if G.out_degree(n) == 0]
    return leaves


def mark_sibling_descendants(tdata, progenitors, key_added="sibling_descendants"):
    """
    Mark cells descending from the sibling clade of each progenitor.

    For each progenitor, the reference node is the progenitor itself if it
    qualified via "bubble_up", otherwise its parent. All leaves below the
    reference node (which include the progenitor's own leaves) are labeled
    with the progenitor node ID. Progenitors whose reference node has more
    than ``5 * fate_descendants`` leaves are skipped. Clones with empty trees
    are skipped. Later progenitors overwrite earlier labels on shared cells.

    Parameters
    ----------
    tdata : td.TreeData
        TreeData with per-clone trees in ``.obst``. ``tdata.obs`` is modified
        in place.
    progenitors : pd.DataFrame
        Progenitor table (e.g. from :func:`get_fate_progenitors`) with columns
        ``clone``, ``node``, ``qualified_via``, ``fate_descendants`` and
        ``time``.
    key_added : str
        Column in ``tdata.obs`` to store the progenitor node ID for marked
        cells (initialized to ``pd.NA``). Defaults to "sibling_descendants".

    Returns
    -------
    None
        Writes ``tdata.obs[key_added]`` and ``tdata.obs["parent_time"]`` (the
        progenitor's ``time`` value) for marked cells.
    """
    tdata.obs[key_added] = pd.NA
    for clone in progenitors["clone"].unique():
        tree = tdata.obst[clone]
        if len(tree) == 0:
            continue
        clone_progenitors = progenitors.query("clone == @clone")
        for _, prog in clone_progenitors.iterrows():
            if prog["qualified_via"] == "bubble_up":
                sibling = prog["node"]
                sibling_leaves = leaves_below(tree, sibling)
            else:
                sibling = list(tree.predecessors(prog["node"]))[0]
                sibling_leaves = leaves_below(tree, sibling)
            if len(sibling_leaves) > 5 * prog["fate_descendants"]:
                continue
            tdata.obs.loc[sibling_leaves, key_added] = prog["node"]
            tdata.obs.loc[sibling_leaves, "parent_time"] = prog["time"]


def bipotency_permutation_test(
    progenitors,
    lineages=("Autonomic", "Sensory"),
    n_permutations=100,
    n_bins=10,
    random_state=None,
    size_key="fate_descendants",
    bipotent_color="#FFAA00",
    ax=None,
):
    """
    Size-stratified permutation test of co-occurrence of two outputs within clades.

    A clade is "bipotent" if it produces both lineages *a* and *b*. The
    observed statistic is the fraction of clades producing *a* or *b* that
    produce both. The null distribution is generated by independently
    permuting the *a* and *b* indicators across clades within quantile bins
    of clade size (*size_key*), preserving the size dependence of each
    output, and recomputing the bipotent fraction. Results are plotted as a
    bar chart (observed vs. permuted; error bars span the 2.5-97.5
    percentiles) overlaid with per-embryo observed percentages. A summary
    table of the exclusive-*a*, exclusive-*b* and bipotent fractions and
    counts among differentiated clades is printed. No p-value is computed.

    Parameters
    ----------
    progenitors : pd.DataFrame
        One row per clade, with boolean columns named by *lineages*, an
        ``embryo`` column, and the *size_key* column.
    lineages : tuple[str, str]
        Names of the two boolean output columns to test. Defaults to
        ("Autonomic", "Sensory").
    n_permutations : int
        Number of size-stratified permutations. Defaults to 100.
    n_bins : int
        Number of quantile bins of *size_key* used for stratification
        (duplicate bin edges are dropped). Defaults to 10.
    random_state : int | None
        Seed for ``np.random.default_rng``.
    size_key : str
        Column used to stratify clades by size. Defaults to "fate_descendants".
    bipotent_color : str
        Bar color for the observed value. Defaults to "#FFAA00".
    ax : matplotlib.axes.Axes | None
        Axes to plot on. If None, a new 1 x 1.5 inch figure is created.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the observed vs. permuted bipotent-percentage plot.
    """
    a, b = lineages
    rng = np.random.default_rng(random_state)

    df = progenitors.copy()
    df["Both"] = df[a] & df[b]
    differentiated = df[df[a] | df[b]]

    summary = pd.DataFrame(
        {
            "category": [a, b, "Both"],
            "fraction": [
                differentiated[a].mean() - differentiated["Both"].mean(),
                differentiated[b].mean() - differentiated["Both"].mean(),
                differentiated["Both"].mean(),
            ],
            "count": [
                differentiated[a].sum() - differentiated["Both"].sum(),
                differentiated[b].sum() - differentiated["Both"].sum(),
                differentiated["Both"].sum(),
            ],
        }
    )

    embryo_fracs = (
        differentiated.groupby("embryo")["Both"].mean().mul(100).reset_index(name="bipotent_pct").assign(permuted=False)
    )

    df["size_bin"] = pd.qcut(df[size_key], n_bins, duplicates="drop")
    rows = [{"permuted": False, "bipotent_frac": differentiated["Both"].mean()}]

    for _ in range(n_permutations):
        pa = df.groupby("size_bin", observed=True)[a].transform(lambda s: rng.permutation(s))
        pb = df.groupby("size_bin", observed=True)[b].transform(lambda s: rng.permutation(s))
        rows.append({"permuted": True, "bipotent_frac": (pa & pb)[pa | pb].mean()})

    bipotent_fracs = pd.DataFrame(rows)
    bipotent_fracs["bipotent_pct"] = 100 * bipotent_fracs["bipotent_frac"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(1, 1.5), dpi=600)
    else:
        fig = ax.figure

    sns.barplot(
        data=bipotent_fracs,
        x="permuted",
        hue="permuted",
        y="bipotent_pct",
        palette=[bipotent_color, "#CCCCCC"],
        errorbar=lambda x: np.percentile(x, [2.5, 97.5]),
        saturation=1,
        legend=False,
        capsize=0.3,
        err_kws={"linewidth": 0.8, "color": "black"},
        ax=ax,
    )

    sns.stripplot(
        data=embryo_fracs,
        x="permuted",
        y="bipotent_pct",
        color="black",
        size=2.5,
        jitter=0,
        ax=ax,
        zorder=10,
    )

    ax.set(ylabel="Bipotent clades (%)", xlabel="")
    ax.set_xticklabels(["Observed", "Permuted"], rotation=45, ha="right")
    print(summary)

    return fig


def progenitor_dispersion_stratified_permutation(
    adata,
    group_col="progenitor",
    strat_col="heart_field",
    embedding_key="X_scvi",
    n_permutations=1000,
    metric="euclidean",
    alternative="greater",
    min_cells=10,
    random_state=0,
):
    """
    Permutation test of clade dispersion in an embedding against stratum-matched random cells.

    For each group (progenitor clade) with at least *min_cells* cells, the
    observed statistic is the mean pairwise distance between its cells in
    ``adata.obsm[embedding_key]``. The group is assigned to its dominant
    stratum (most frequent value of *strat_col* among its cells), and a null
    distribution is built by repeatedly drawing the same number of cells
    without replacement from all cells in that stratum. Null distributions
    are cached and reused per (stratum, group size). Groups whose dominant
    stratum contains fewer cells than the group are skipped. P-values use the
    ``(k + 1) / (n_permutations + 1)`` correction and are adjusted with
    Benjamini-Hochberg FDR. A dense all-by-all distance matrix is computed
    up front, so memory scales quadratically with ``adata.n_obs``.

    Parameters
    ----------
    adata : AnnData | td.TreeData
        Cells to analyze.
    group_col : str
        Column in ``adata.obs`` defining the groups (clades). Defaults to
        "progenitor".
    strat_col : str
        Column in ``adata.obs`` defining the strata used to build the null.
        Defaults to "heart_field".
    embedding_key : str
        Key in ``adata.obsm`` of the embedding. Defaults to "X_scvi".
    n_permutations : int
        Number of random draws per null distribution. Defaults to 1000.
    metric : str
        Distance metric passed to ``scipy.spatial.distance.pdist``. Defaults
        to "euclidean".
    alternative : {"greater", "less", "two-sided"}
        "greater" tests for more dispersion than the null, "less" for less
        dispersion (tighter clustering), and "two-sided" for an absolute
        deviation from the null mean. Defaults to "greater".
    min_cells : int
        Minimum number of cells for a group to be tested. Defaults to 10.
    random_state : int | None
        Seed for ``np.random.default_rng``. Defaults to 0.

    Returns
    -------
    pd.DataFrame
        One row per tested group, indexed by ``progenitor``, with columns:
          n_cells     - number of cells in the group
          heart_field - dominant stratum (named "heart_field" regardless of
                        *strat_col*)
          observed    - observed mean pairwise distance
          null_mean   - mean of the null distribution
          null_std    - standard deviation of the null (ddof=1)
          z_score     - (observed - null_mean) / null_std (NaN if null_std is 0)
          p_value     - permutation p-value
          p_adj       - Benjamini-Hochberg adjusted p-value
        Empty if no group is tested.

    Raises
    ------
    ValueError
        If *alternative* is not one of "greater", "less", "two-sided"
        (raised when the first eligible group is tested).
    """
    rng = np.random.default_rng(random_state)

    X = adata.obsm[embedding_key]
    groups = adata.obs[group_col].to_numpy()
    strata = adata.obs[strat_col].to_numpy()

    # Precompute full pairwise distance matrix once
    D = squareform(pdist(X, metric=metric))

    def mean_upper_triangle(submatrix):
        """Mean of the strictly upper-triangular entries of a square distance submatrix."""
        n = submatrix.shape[0]
        return submatrix[np.triu_indices(n, k=1)].mean()

    # Cache cell indices for each heart_field
    heart_field_to_idx = {hf: np.where(strata == hf)[0] for hf in pd.unique(strata)}

    # Cache null distributions for each (heart_field, n_cells)
    null_cache = {}

    results = []

    for prog in pd.unique(groups):
        idx = np.where(groups == prog)[0]
        n = len(idx)

        if n < min_cells:
            continue

        # dominant heart_field for this progenitor
        prog_hf_counts = pd.Series(strata[idx]).value_counts()
        dominant_hf = prog_hf_counts.idxmax()

        pool_idx = heart_field_to_idx[dominant_hf]
        if len(pool_idx) < n:
            continue

        # observed mean pairwise distance
        obs = mean_upper_triangle(D[np.ix_(idx, idx)])

        cache_key = (dominant_hf, n)

        if cache_key not in null_cache:
            null = np.empty(n_permutations, dtype=float)
            for i in range(n_permutations):
                perm_idx = rng.choice(pool_idx, size=n, replace=False)
                null[i] = mean_upper_triangle(D[np.ix_(perm_idx, perm_idx)])
            null_cache[cache_key] = null
        else:
            null = null_cache[cache_key]

        null_mean = null.mean()
        null_std = null.std(ddof=1)

        if alternative == "greater":
            p = (np.sum(null >= obs) + 1) / (n_permutations + 1)
        elif alternative == "less":
            p = (np.sum(null <= obs) + 1) / (n_permutations + 1)
        elif alternative == "two-sided":
            p = (np.sum(np.abs(null - null_mean) >= np.abs(obs - null_mean)) + 1) / (n_permutations + 1)
        else:
            raise ValueError("alternative must be one of: 'greater', 'less', 'two-sided'")

        z = np.nan if null_std == 0 else (obs - null_mean) / null_std

        results.append(
            {
                "progenitor": prog,
                "n_cells": n,
                "heart_field": dominant_hf,
                "observed": obs,
                "null_mean": null_mean,
                "null_std": null_std,
                "z_score": z,
                "p_value": p,
            }
        )

    df = pd.DataFrame(results)

    if df.empty:
        return df.set_index("progenitor")

    df["p_adj"] = multipletests(df["p_value"], method="fdr_bh")[1]
    return df.set_index("progenitor")
