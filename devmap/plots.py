"""Reusable plot functions for devmap figures."""

import matplotlib.pyplot as plt
import seaborn as sns
import pycea as py
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as sch

from .config import edit_palette
from .linkage import symmetrize_with_mean

def barplot_with_points(
    data,
    x,
    y,
    hue=None,
    order=None,
    hue_order=None,
    palette=None,
    ax=None,
    point_size=3,
    jitter=0.15,
    errorbar="se",
    capsize=0.3,
):
    """Bar plot with individual data points overlaid.

    Combines a seaborn barplot (SE error bars) with a stripplot of the raw
    observations, which is a common pattern for small-n categorical data.

    Parameters
    ----------
    data : DataFrame
    x : str
        Column for the x-axis categories.
    y : str
        Column for the y-axis values.
    hue : str, optional
        Column for color grouping. Defaults to x if not provided.
    order : list, optional
        Order of x categories.
    palette : dict or list, optional
        Colors for hue levels.
    ax : Axes, optional
        Axes to draw on. Creates new axes if None.
    point_size : float
        Size of individual data points.
    jitter : float
        Horizontal jitter for strip points.
    errorbar : str or tuple
        Error bar type passed to seaborn (default "se").
    capsize : float
        Width of error bar caps.

    Returns
    -------
    ax : Axes
    """
    if hue is None:
        hue = x
    if ax is None:
        _, ax = plt.subplots()

    data = data.sort_values(hue)

    sns.barplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        order=order,
        palette=palette,
        hue_order=hue_order,
        saturation=1,
        edgecolor="black",
        linewidth=0.8,
        legend=False,
        errorbar=errorbar,
        capsize=capsize,
        err_kws={"linewidth": 0.8,"color":"black"},
        ax=ax,
    )
    sns.stripplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        order=order,
        hue_order=hue_order,
        dodge=True,
        jitter=jitter,
        palette="dark:black",
        size=point_size,
        legend=False,
        ax=ax,
    )
    return ax

def plot_grouped_characters(tdata,ax = None,width = .1,label = False,offset = .05):
    """Plot allele table grouped by integration"""
    if ax is None:
        ax = plt.gca()
    tdata.obs = tdata.obs.merge(tdata.obsm["characters"].astype(str),left_index=True,right_index=True)
    ids = tdata.obsm["characters"].columns.str.split("-").str[0].str.replace("intID","").unique()
    ids = sorted(ids,key = lambda x: int(x))
    for i,id in enumerate(ids):
        gap = offset if i == 0 else width/2
        py.pl.annotation(tdata,keys=[f"intID{id}-RNF2",f"intID{id}-HEK3",f"intID{id}-EMX1"],border_width=.3,
                            label = id if label else False,width=width,gap = gap,palette = edit_palette,ax = ax, legend = False)
    tdata.obs = tdata.obs.drop(columns = tdata.obsm["characters"].columns)
    ax.tick_params(axis='x', pad=0)

def plot_program_lineplot(df_top, df_rest, y, ax, palette, x="time", hue="program"):
    df_top[hue] = df_top[hue].astype(str)
    df_rest[hue] = df_rest[hue].astype(str)
    # Background (gray)
    sns.lineplot(
        data=df_rest,
        x=x,
        y=y,
        hue=hue,
        units=hue,
        estimator=None,
        palette=["lightgray"],
        linewidth=0.5,
        legend=False,
        ax=ax
    )
    # Top programs (colored)
    sns.lineplot(
        data=df_top,
        x=x,
        y=y,
        hue=hue,
        palette=palette,
        linewidth=1.5,
        legend=True,
        ax=ax
    )
    plt.xticks([0, 2, 4, 6, 8]);
    return ax


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
    show_var = True,
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
    if show_var:
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

    # ── axes ──────────────────────────────────────────────────────────
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # ── lower-triangle heatmap ────────────────────────────────────────
    mask_upper = np.triu(np.ones_like(df, dtype=bool), k=0)
    sns.heatmap(
        df,
        xticklabels=df.columns,
        yticklabels=df.index,
        mask=mask_upper if show_var else None,
        cmap=cmap,
        center=center,
        vmin=vmin,
        vmax=vmax,
        cbar=False,
        ax=ax,
    )
    ax.tick_params(axis="both", labelsize=tick_labelsize)

    # ── upper-triangle scatter ────────────────────────────────────────
    if show_var:
        var_plot = var_mat.loc[df.index, df.columns]
        pval_plot = pval_mat.loc[df.index, df.columns]

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
