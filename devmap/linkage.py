import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as sch
import seaborn as sns

def symmetrize_with_mean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace entries above/below diagonal with their mean,
    so the result is symmetric.
    The diagonal itself is left unchanged.
    """
    if df.shape[0] != df.shape[1]:
        raise ValueError("DataFrame must be square.")

    arr = df.values.astype(float).copy()
    n = arr.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            m = (arr[i, j] + arr[j, i]) / 2.0
            arr[i, j] = m
            arr[j, i] = m
    return pd.DataFrame(arr, index=df.index, columns=df.columns)


def _cluster_order(similarity_df: pd.DataFrame, method: str = "ward") -> np.ndarray:
    """Return optimal leaf order for a square similarity matrix."""
    S = similarity_df.fillna(0)
    D = S.max().max() - S
    np.fill_diagonal(D.values, 0)
    dcond = ssd.squareform(D.values, checks=False)
    Z = sch.linkage(dcond, method=method)
    Z_opt = sch.optimal_leaf_ordering(Z, dcond)
    return sch.leaves_list(Z_opt)


def plot_linkage_heatmap(
    embryo_linkage_stats: pd.DataFrame,
    symmetrize: str | None = "mean",
    ax: plt.Axes | None = None,
    cmap: str = "RdBu_r",
    center: float = 0,
    vmin: float = -2,
    vmax: float = 2,
    var_vmax: float = 0.2,
    order: list | pd.Index | np.ndarray | None = None,
    scatter_size_range: tuple[float, float] = (2, 10),
    tick_labelsize: float = 5,
    order_method: str = "ward",
    figsize: tuple[float, float] = (6.2, 6.2),
    dpi: int = 600,
) -> tuple[plt.Axes, pd.Index]:
    """Plot a lower-triangle heatmap with upper-triangle variance / p-value scatter.

    Parameters
    ----------
    embryo_linkage_stats : DataFrame with columns
        ``source``, ``target``, ``norm_value``, ``norm_value_var``, ``p_value``.
    symmetrize : str or None
        How to symmetrize the linkage matrix. Options are "mean" or None.
    ax : matplotlib Axes, optional
        If *None* a new figure and axes are created.
    cmap : colormap for the heatmap.
    center, vmin, vmax : heatmap colour-scale parameters.
    var_vmax : upper bound for variance colour normalisation.
    order : list-like or None
        Optional order of rows/columns. If None, the order is determined by clustering.
    scatter_size_range : (min, max) marker sizes for the p-value scatter.
    tick_labelsize : font size for tick labels.
    figsize, dpi : used only when *ax* is None.

    Returns
    -------
    ax : matplotlib Axes
    ordered_index : pd.Index
        Row / column labels in clustered display order (top-to-bottom,
        left-to-right). Use this to align marginal colour strips.
    """
    # ── pivot & symmetrize ────────────────────────────────────────────
    if symmetrize == "mean":
        symmetrize_fn = symmetrize_with_mean
    elif symmetrize is None:
        symmetrize_fn = lambda df: df
    else:
        raise ValueError(f"Invalid symmetrize option: {symmetrize!r}")
    norm_linkage = (
        embryo_linkage_stats
        .pivot(index="source", columns="target", values="norm_value")
        .pipe(symmetrize_fn)
    )
    var_mat = (
        embryo_linkage_stats
        .pivot(index="source", columns="target", values="norm_value_var")
        .fillna(0)
        .pipe(symmetrize_fn)
    )
    pval_mat = (
        embryo_linkage_stats
        .pivot(index="source", columns="target", values="p_value")
        .fillna(1)
        .pipe(symmetrize_fn)
    )

    # ── cluster & reorder ─────────────────────────────────────────────
    if order is not None:
        df = norm_linkage.loc[order, order]
    else:
        cluster_order = _cluster_order(norm_linkage, method=order_method)
        rev_order = list(reversed(cluster_order))
        df = norm_linkage.iloc[rev_order, rev_order]

    var_plot = var_mat.loc[df.index, df.columns]
    pval_plot = pval_mat.loc[df.index, df.columns]

    # ── axes ──────────────────────────────────────────────────────────
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # ── lower-triangle heatmap ────────────────────────────────────────
    mask_upper = np.triu(np.ones_like(df, dtype=bool), k=0)
    sns.heatmap(
        df,
        xticklabels=df.columns,
        yticklabels=df.index,
        mask=mask_upper,
        cmap=cmap,
        center=center,
        vmin=vmin,
        vmax=vmax,
        cbar=False,
        ax=ax,
    )
    ax.tick_params(axis="both", labelsize=tick_labelsize)

    # ── upper-triangle scatter ────────────────────────────────────────
    nrows, ncols = df.shape
    ii, jj = np.where(np.triu(np.ones((nrows, ncols), dtype=bool), k=1))
    x, y = jj + 0.5, ii + 0.5

    var_vals = var_plot.values[ii, jj]
    pval_vals = pval_plot.values[ii, jj]

    # p-value → marker size (smaller p → larger circle)
    s_min, s_max = scatter_size_range
    raw_sizes = -np.log10(np.clip(pval_vals, 1e-10, 1))
    size_range = raw_sizes.max() - raw_sizes.min()
    if size_range > 0:
        sizes = s_min + (raw_sizes - raw_sizes.min()) / size_range * (s_max - s_min)
    else:
        sizes = np.full_like(raw_sizes, (s_min + s_max) / 2)

    ax.scatter(
        x, y,
        s=sizes,
        c=var_vals,
        cmap=cm.get_cmap("Grays"),
        norm=mcolors.Normalize(vmin=0, vmax=var_vmax),
        marker="o",
        edgecolors="black",
        linewidths=0.3,
    )

    ax.set_xlabel("")
    ax.set_ylabel("")

    return ax, df.columns


# ── marginal colour strips ────────────────────────────────────────────────────

def draw_marginal_strip(
    ax: plt.Axes,
    values: pd.Series,
    palette: dict,
    ordered_index: pd.Index,
    orientation: str = "col",
    fallback_color: str = "lightgrey",
) -> None:
    """Draw a colour strip aligned to *ordered_index*.

    Parameters
    ----------
    ax : matplotlib Axes.
    values : Series mapping each label in *ordered_index* to a category.
    palette : dict mapping category → matplotlib colour.
    ordered_index : label order returned by :func:`plot_linkage_heatmap`.
    orientation : ``"col"`` for a horizontal strip (above/below the heatmap)
        or ``"row"`` for a vertical strip (left/right of the heatmap).
    fallback_color : colour used when a label has no category.
    """
    colors = values.reindex(ordered_index).map(palette).fillna(fallback_color)
    rgba = np.array([mcolors.to_rgba(c) for c in colors])

    if orientation == "col":
        ax.imshow(rgba[np.newaxis, :, :], aspect="auto")
    elif orientation == "row":
        ax.imshow(rgba[:, np.newaxis, :], aspect="auto")
    else:
        raise ValueError(f"orientation must be 'col' or 'row', got {orientation!r}")

    ax.set_axis_off()