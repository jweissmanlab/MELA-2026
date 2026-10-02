"""Reusable plot functions for devmap figures."""

from collections.abc import Mapping, Sequence

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycea as py
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import to_hex, to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path

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
        err_kws={"linewidth": 0.8, "color": "black"},
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


def plot_grouped_characters(tdata, ax=None, width=0.1, label=False, offset=0.05):
    """Plot the allele table as annotation strips grouped by integration.

    For each integration ID (parsed from ``obsm["characters"]`` columns named
    ``intID<id>-<site>``, sorted numerically), draws a ``py.pl.annotation`` block
    with the RNF2, HEK3, and EMX1 sites colored by ``edit_palette``. The
    character columns are temporarily merged into ``tdata.obs`` and removed
    afterwards.

    Parameters
    ----------
    tdata : TreeData
        TreeData with an ``obsm["characters"]`` allele table.
    ax : Axes, optional
        Axes to draw on (typically one already holding a tree plot). Defaults to
        the current axes.
    width : float
        Width of each site strip.
    label : bool
        If True, label each integration block with its integration ID.
    offset : float
        Gap before the first integration block; subsequent blocks are separated
        by ``width / 2``.

    Returns
    -------
    None
    """
    if ax is None:
        ax = plt.gca()
    tdata.obs = tdata.obs.merge(tdata.obsm["characters"].astype(str), left_index=True, right_index=True)
    ids = tdata.obsm["characters"].columns.str.split("-").str[0].str.replace("intID", "").unique()
    ids = sorted(ids, key=lambda x: int(x))
    for i, id in enumerate(ids):
        gap = offset if i == 0 else width / 2
        py.pl.annotation(
            tdata,
            keys=[f"intID{id}-RNF2", f"intID{id}-HEK3", f"intID{id}-EMX1"],
            border_width=0.3,
            label=id if label else False,
            width=width,
            gap=gap,
            palette=edit_palette,
            ax=ax,
            legend=False,
        )
    tdata.obs = tdata.obs.drop(columns=tdata.obsm["characters"].columns)
    ax.tick_params(axis="x", pad=0)


def plot_program_lineplot(df_top, df_rest, y, ax, palette, x="time", hue="program"):
    """Line plot of gene-program scores with highlighted top programs.

    Background programs are drawn as thin light-gray lines (one per program, no
    aggregation); top programs are drawn on top in color with a legend. The
    ``hue`` column of both input DataFrames is cast to ``str`` in place. X ticks
    are set to ``[0, 2, 4, 6, 8]`` on the current pyplot axes.

    Parameters
    ----------
    df_top : DataFrame
        Long-form data for the highlighted programs.
    df_rest : DataFrame
        Long-form data for the background programs.
    y : str
        Column for the y-axis values.
    ax : Axes
        Axes to draw on.
    palette : dict or list
        Colors for the highlighted programs.
    x : str
        Column for the x-axis values.
    hue : str
        Column identifying programs.

    Returns
    -------
    ax : Axes
    """
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
        ax=ax,
    )
    # Top programs (colored)
    sns.lineplot(data=df_top, x=x, y=y, hue=hue, palette=palette, linewidth=1.5, legend=True, ax=ax)
    plt.xticks([0, 2, 4, 6, 8])
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


def _upper_tri_df_to_matrix(
    df: pd.DataFrame,
    value_col: str = "value",
    fill_diagonal: bool | None = None,
) -> pd.DataFrame:
    """Convert a long-form DataFrame of upper-triangle values to a square matrix."""
    mat = df.pivot(index="source", columns="target", values=value_col)

    # Ensure all labels are present on both axes
    labels = mat.index.union(mat.columns)
    mat = mat.reindex(index=labels, columns=labels)

    values = mat.to_numpy()

    # Copy upper triangle to lower triangle
    i_lower = np.tril_indices_from(values, k=-1)
    values[i_lower] = values.T[i_lower]

    if fill_diagonal is not None:
        np.fill_diagonal(values, fill_diagonal)

    return pd.DataFrame(values, index=labels, columns=labels)


def plot_linkage_heatmap(
    embryo_linkage_stats: pd.DataFrame,
    symmetrize: str | None = None,
    ax: plt.Axes | None = None,
    show_var=True,
    cmap: str = "RdBu_r",
    center: float = 0,
    vmin: float = -2,
    vmax: float = 2,
    var_vmax: float = 0.2,
    order: list | pd.Index | np.ndarray | None = None,
    scatter_size_range: tuple[float, float] = (2, 10),
    tick_labelsize: float = 5,
    order_method: str = "ward",
    lower_diagonal_only: bool = False,
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
    show_var : bool
        If True, draw the upper-triangle scatter (color = ``norm_value_var``,
        size = -log10 ``p_value``) and mask the upper triangle of the heatmap.
    cmap : colormap for the heatmap.
    center, vmin, vmax : heatmap colour-scale parameters.
    var_vmax : upper bound for variance colour normalisation.
    order : list-like or None
        Optional order of rows/columns. If None, the order is determined by clustering.
    scatter_size_range : (min, max) marker sizes for the p-value scatter.
    tick_labelsize : font size for tick labels.
    order_method : linkage method passed to scipy for clustering when *order* is None.
    lower_diagonal_only : if True, mask the upper triangle even when *show_var* is False.
    figsize, dpi : used only when *ax* is None.

    Returns
    -------
    ax : matplotlib Axes
    ordered_index : pd.Index
        Row / column labels in clustered display order (top-to-bottom,
        left-to-right). Use this to align marginal colour strips.

    Raises
    ------
    ValueError
        If *symmetrize* is not "mean" or None.
    """
    # ── pivot & symmetrize ────────────────────────────────────────────
    if symmetrize == "mean":
        symmetrize_fn = symmetrize_with_mean
    elif symmetrize is None:
        pass
    else:
        raise ValueError(f"Invalid symmetrize option: {symmetrize!r}")
    if symmetrize is not None:
        embryo_linkage_stats = (
            embryo_linkage_stats.pivot(index="source", columns="target", values="norm_value")
            .pipe(symmetrize_fn)
            .reset_index()
            .melt(id_vars="source", var_name="target", value_name="norm_value")
        )
    else:
        norm_linkage = _upper_tri_df_to_matrix(embryo_linkage_stats, value_col="norm_value", fill_diagonal=0)
    if show_var:
        if symmetrize is not None:
            var_mat = (
                embryo_linkage_stats.pivot(index="source", columns="target", values="norm_value_var")
                .fillna(0)
                .pipe(symmetrize_fn)
            )
            pval_mat = (
                embryo_linkage_stats.pivot(index="source", columns="target", values="p_value")
                .fillna(1)
                .pipe(symmetrize_fn)
            )
        else:
            var_mat = _upper_tri_df_to_matrix(embryo_linkage_stats, value_col="norm_value_var", fill_diagonal=0)
            pval_mat = _upper_tri_df_to_matrix(embryo_linkage_stats, value_col="p_value", fill_diagonal=1)

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
        mask=mask_upper if (show_var or lower_diagonal_only) else None,
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
            x,
            y,
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
    ax.tick_params(axis="both", which="both", length=0)

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

    Raises
    ------
    ValueError
        If *orientation* is not "col" or "row".
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


def plot_restriction_sankey(
    transitions: pd.DataFrame,
    palette: Mapping[str, str] | None = None,
    order: Sequence[str] | None = None,
    *,
    ax: Axes | None = None,
    normalize: bool = False,
    gap: float = 0.02,
    node_width: float = 0.1,
    ribbon_alpha: float = 0.45,
    xlabel: str = "time",
) -> tuple[Figure, Axes]:
    """
    Plot transitions between extant branches as a Sankey diagram.

    Parameters
    ----------
    transitions
        Output of ``extant_transition_count``. Must contain
        ``source_time``, ``target_time``, ``source``, ``target``, and
        ``count``.
    palette
        Optional mapping from category names to colors. Missing categories
        receive colors from Matplotlib's default color cycle.
    order
        Optional bottom-to-top category order. Present categories omitted
        from ``order`` are appended alphabetically.
    ax
        Existing axes. If None, new axes are created using Matplotlib's
        default figure size.
    normalize
        If True, normalize bar heights so that categories at each timepoint
        sum to one.
    gap
        Gap between category bars. When ``normalize=False``, this is a
        fraction of the maximum total number of extant branches. When
        ``normalize=True``, it is in normalized y-axis units.
    node_width
        Width of node bars in x-axis units.
    ribbon_alpha
        Ribbon opacity.
    xlabel
        X-axis label.

    Returns
    -------
    figure, axes
        Matplotlib figure and axes.
    """
    required = {
        "source_time",
        "target_time",
        "source",
        "target",
        "count",
    }
    missing = required.difference(transitions.columns)
    if missing:
        raise ValueError(f"transitions is missing required columns: {sorted(missing)}")

    links = transitions[["source_time", "target_time", "source", "target", "count"]].copy()

    links = links.loc[links["count"].gt(0)]
    links["source"] = links["source"].astype(str)
    links["target"] = links["target"].astype(str)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if links.empty:
        return fig, ax

    links = links.groupby(
        ["source_time", "source", "target_time", "target"],
        observed=True,
        as_index=False,
    )["count"].sum()

    times = sorted(set(links["source_time"]).union(links["target_time"]))
    time_rank = {time: i for i, time in enumerate(times)}

    present = set(links["source"]).union(links["target"])

    if order is None:
        categories = sorted(present)
    else:
        requested = [str(category) for category in order]
        categories = [category for category in requested if category in present]
        categories.extend(sorted(present.difference(categories)))

    category_rank = {category: rank for rank, category in enumerate(categories)}

    colors = dict(palette or {})
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, category in enumerate(categories):
        colors.setdefault(
            category,
            default_colors[i % len(default_colors)],
        )

    # Keep ribbon stacking consistent across nodes.
    links["_source_time_rank"] = links["source_time"].map(time_rank)
    links["_target_time_rank"] = links["target_time"].map(time_rank)
    links["_source_rank"] = links["source"].map(category_rank)
    links["_target_rank"] = links["target"].map(category_rank)

    links = links.sort_values(
        [
            "_source_time_rank",
            "_source_rank",
            "_target_time_rank",
            "_target_rank",
        ]
    ).reset_index(drop=True)

    outgoing = links.groupby(["source_time", "source"], observed=True)["count"].sum().to_dict()
    incoming = links.groupby(["target_time", "target"], observed=True)["count"].sum().to_dict()

    # Use outgoing counts at intermediate timepoints so the outgoing ribbons
    # exactly reproduce the corresponding bar height. The final timepoint has
    # no outgoing links and therefore uses incoming counts.
    node_counts: dict[tuple[object, str], float] = {}

    for time in times:
        for category in categories:
            key = (time, category)
            node_counts[key] = outgoing[key] if key in outgoing else incoming.get(key, 0)

    time_totals = {time: sum(node_counts[(time, category)] for category in categories) for time in times}

    if normalize:
        for time in times:
            total = time_totals[time]
            if total > 0:
                for category in categories:
                    node_counts[(time, category)] /= total

        absolute_gap = gap
    else:
        max_total = max(time_totals.values(), default=0)
        absolute_gap = gap * max_total

    node_positions: dict[
        tuple[object, str],
        tuple[float, float],
    ] = {}

    for time in times:
        y = 0.0

        for category in categories:
            size = node_counts[(time, category)]
            node_positions[(time, category)] = (y, y + size)

            if size > 0:
                y += size + absolute_gap

    source_offsets = {key: bounds[0] for key, bounds in node_positions.items()}
    target_offsets = {key: bounds[0] for key, bounds in node_positions.items()}

    # Scale each side independently so all ribbons exactly fill the bar at
    # both their source and target endpoints.
    source_scale: dict[tuple[object, str], float] = {}
    target_scale: dict[tuple[object, str], float] = {}

    for key, (y0, y1) in node_positions.items():
        bar_height = y1 - y0
        outgoing_total = outgoing.get(key, 0)
        incoming_total = incoming.get(key, 0)

        source_scale[key] = bar_height / outgoing_total if outgoing_total > 0 else 1.0
        target_scale[key] = bar_height / incoming_total if incoming_total > 0 else 1.0

    for row in links.itertuples(index=False):
        source_key = (row.source_time, row.source)
        target_key = (row.target_time, row.target)

        source_width = row.count * source_scale[source_key]
        y0a = source_offsets[source_key]
        y0b = y0a + source_width
        source_offsets[source_key] = y0b

        target_width = row.count * target_scale[target_key]
        y1a = target_offsets[target_key]
        y1b = y1a + target_width
        target_offsets[target_key] = y1b

        x0 = time_rank[row.source_time]
        x1 = time_rank[row.target_time]
        dx = 0.45 * (x1 - x0)

        vertices = [
            (x0, y0a),
            (x0 + dx, y0a),
            (x1 - dx, y1a),
            (x1, y1a),
            (x1, y1b),
            (x1 - dx, y1b),
            (x0 + dx, y0b),
            (x0, y0b),
            (x0, y0a),
        ]
        codes = [
            Path.MOVETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.LINETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.CLOSEPOLY,
        ]

        ax.add_patch(
            PathPatch(
                Path(vertices, codes),
                facecolor=colors[row.source],
                edgecolor="none",
                alpha=ribbon_alpha,
            )
        )

    half_width = node_width / 2

    for (time, category), (y0, y1) in node_positions.items():
        if y1 <= y0:
            continue

        x = time_rank[time]

        ax.fill_between(
            [x - half_width, x + half_width],
            y0,
            y1,
            color=colors[category],
            edgecolor="black",
            linewidth=0.5,
            zorder=2,
        )

    legend_handles = [
        Patch(
            facecolor=colors[category],
            edgecolor="black",
            linewidth=0.5,
            label=category,
        )
        for category in categories
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        frameon=False,
    )

    ax.set_xticks(range(len(times)))
    ax.set_xticklabels(times, rotation=90)
    ax.set_yticks([])
    ax.set_xlabel(xlabel)
    ax.set_xlim(-0.1, len(times) - 0.9)
    ax.margins(y=0.02)

    return fig, ax


def simplex_to_hex(w, gray="#D3D3D3"):
    """Map 3-component simplex weights to hex colors.

    Weights are normalized to sum to one and used to linearly mix a red
    (``#CD2626``), blue (``#1874CD``), and yellow (``#FFE600``) basis.

    Parameters
    ----------
    w : array-like
        Weights of shape ``(3,)`` or ``(n, 3)``.
    gray : str
        Color assigned to rows whose weights sum to zero.

    Returns
    -------
    str or ndarray of str
        A single hex color if one row is given, otherwise an array of hex colors.
    """
    BASIS = np.array([to_rgb(h) for h in ("#CD2626", "#1874CD", "#FFE600")])
    w = np.asarray(w, dtype=float)
    if w.ndim == 1:
        w = w[np.newaxis, :]
    sums = w.sum(axis=1, keepdims=True)
    zero_mask = sums.ravel() == 0
    safe_sums = np.where(sums == 0, 1, sums)
    normed = w / safe_sums
    rgb = normed @ BASIS
    hexes = np.array([to_hex(row) for row in rgb])
    hexes[zero_mask] = gray
    return hexes if len(hexes) > 1 else hexes[0]


def hex_to_simplex(hex_color):
    """Recover simplex weights from a hex color by least-squares against the basis.

    Inverse of :func:`simplex_to_hex`: solves for weights on the red/blue/yellow
    basis, clips negatives to zero, and renormalizes to sum to one.

    Parameters
    ----------
    hex_color : str
        Hex color, ideally one produced by :func:`simplex_to_hex`.

    Returns
    -------
    ndarray
        Weights of shape ``(3,)`` summing to one.
    """
    BASIS = np.array([to_rgb(h) for h in ("#CD2626", "#1874CD", "#FFE600")])
    rgb = np.array(to_rgb(hex_color))
    # Solve: w @ BASIS ≈ rgb, with w >= 0
    # Use least-squares (exact for colors generated by simplex_to_hex)
    w, *_ = np.linalg.lstsq(BASIS.T, rgb, rcond=None)
    w = np.clip(w, 0, None)
    w /= w.sum()
    return w


def simplex_sort_key(hex_color, gray="#D3D3D3"):
    """
    Sort key giving the angular position of a color on the simplex.

    The color is converted to simplex weights with :func:`hex_to_simplex`,
    projected onto a 2D triangle, and keyed by its angle around the centroid.
    Gray colors sort to the bottom (lowest key).

    Parameters
    ----------
    hex_color : str
        Hex color to sort.
    gray : str
        Color treated as gray (case-insensitive match).

    Returns
    -------
    tuple of float
        One-element tuple with the angle in radians, or ``(-inf,)`` for gray.
    """
    if hex_color.lower() == gray.lower():
        return (-np.inf,)
    w = hex_to_simplex(hex_color)
    vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
    x, y = w @ vertices
    cx, cy = vertices.mean(axis=0)
    return (np.arctan2(y - cy, x - cx),)


def plot_area(
    df,
    x="time",
    color="color",
    value="n",
    ax=None,
    sort_fn=None,
    percent=True,
):
    """Stacked area plot of counts per color over time.

    Values are summed per ``(x, color)`` and stacked; the values of the
    ``color`` column are used directly as fill (and edge) colors.

    Parameters
    ----------
    df : DataFrame
        Long-form data.
    x : str
        Column for the x-axis (e.g. time).
    color : str
        Column of matplotlib colors defining the stacked groups.
    value : str
        Column of values to sum.
    ax : Axes
        Axes to draw on. Required; no axes are created if None.
    sort_fn : callable, optional
        Key function applied to colors to set the stacking order (e.g.
        :func:`simplex_sort_key`).
    percent : bool
        If True, normalize each x value to percentages (y limits 0-100, label
        "Extant cells (%)"); otherwise plot raw sums ("Number of cells").

    Returns
    -------
    None
    """
    pivot = df.pivot_table(
        index=x,
        columns=color,
        values=value,
        aggfunc="sum",
        fill_value=0,
    )

    if percent:
        plot_df = pivot.div(pivot.sum(axis=1), axis=0) * 100
    else:
        plot_df = pivot

    plot_df = plot_df.sort_index()

    if sort_fn is not None:
        sorted_cols = sorted(plot_df.columns, key=sort_fn)
        plot_df = plot_df[sorted_cols]

    polys = ax.stackplot(
        plot_df.index,
        *[plot_df[col] for col in plot_df.columns],
        colors=plot_df.columns,
        linewidth=0.05,
    )

    # Match edge colors to fill colors to hide anti-aliasing gaps
    for poly, col in zip(polys, plot_df.columns, strict=False):
        poly.set_edgecolor(col)

    if percent:
        ax.set_ylim(0, 100)
        ax.set_ylabel("Extant cells (%)")
    else:
        ax.set_ylabel("Number of cells")


def plot_simplex_triangle(resolution=700, figsize=(1, 1), ax=None):
    """Draw a color-key triangle for :func:`simplex_to_hex`.

    Each point inside the triangle is colored by its barycentric weights; the
    top, bottom-left, and bottom-right vertices correspond to the first
    (red), second (blue), and third (yellow) weights.

    Parameters
    ----------
    resolution : int
        Number of pixels along the x-axis of the rendered image.
    figsize : tuple of float
        Figure size, used only when ``ax`` is None.
    ax : Axes, optional
        Axes to draw on. Creates a new figure and axes if None.

    Returns
    -------
    ax : Axes
    """

    def cartesian_to_barycentric(xy, V):
        """Convert Cartesian points ``xy`` (..., 2) to barycentric coordinates w.r.t. triangle ``V``."""
        V = np.asarray(V, dtype=float)
        A = np.array(
            [
                [V[0, 0], V[1, 0], V[2, 0]],
                [V[0, 1], V[1, 1], V[2, 1]],
                [1.0, 1.0, 1.0],
            ]
        )
        Ainv = np.linalg.inv(A)
        pts = np.asarray(xy, dtype=float).reshape(-1, 2)
        aug = np.column_stack([pts, np.ones(len(pts))])
        bary = aug @ Ainv.T
        return bary.reshape(*xy.shape[:-1], 3)

    V = np.array(
        [
            [0.0, 1.0],
            [-np.sqrt(3) / 2, -0.5],
            [np.sqrt(3) / 2, -0.5],
        ]
    )

    xmin, xmax = V[:, 0].min(), V[:, 0].max()
    ymin, ymax = V[:, 1].min(), V[:, 1].max()

    nx = resolution
    ny = int(resolution * (ymax - ymin) / (xmax - xmin))

    X, Y = np.meshgrid(np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny))
    bary = cartesian_to_barycentric(np.stack([X, Y], axis=-1), V)
    inside = np.all(bary >= -1e-10, axis=-1)

    img = np.ones((ny, nx, 4))
    img[..., 3] = 0.0

    # Use simplex_to_hex, then convert back to RGB for the image
    w = bary[inside]
    w = np.clip(w, 0, None)
    hex_arr = simplex_to_hex(w)
    rgb = np.array([to_rgb(h) for h in hex_arr])

    img[inside, :3] = rgb
    img[inside, 3] = 1.0

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(img, origin="lower", extent=[xmin, xmax, ymin, ymax], interpolation="bilinear")
    tri = np.vstack([V, V[0]])
    ax.plot(tri[:, 0], tri[:, 1], color="black", lw=1)
    ax.set_aspect("equal")
    ax.set_xlim(xmin - 0.03, xmax + 0.03)
    ax.set_ylim(ymin - 0.03, ymax + 0.03)
    ax.axis("off")
    return ax


def plot_progenitor_output(
    obs,
    progenitor_col,
    fate_col,
    fate_order,
    min_cells=2,
    row_order="cluster",
    row_group_fn=None,
    sort_col=None,
    sort_ascending=False,
    figsize=(2, 2),
    dpi=600,
    cmap=None,
    vmax=None,
    ylim=None,
    yticks=None,
    xlabel="Progenitors",
):
    """
    Plot total progenitor output above a heatmap of descendant fate fractions.

    Parameters
    ----------
    obs : pd.DataFrame
        Cell-level metadata.
    progenitor_col : str
        Column identifying progenitors.
    fate_col : str
        Column containing descendant fate/category.
    fate_order : list[str]
        Order of rows in the heatmap.
    min_cells : int
        Minimum total output required to retain a progenitor.
    row_order : {"cluster", "sort"}
        How progenitors are ordered.
    row_group_fn : callable, optional
        Function taking the fraction dataframe and returning a Series used
        to group/sort rows. Used when row_order="sort".
    sort_col : str, optional
        Fraction column used for within-group sorting.
    sort_ascending : bool
        Direction of within-group sorting.
    figsize : tuple
        Figure size.
    dpi : int
        Figure DPI.
    cmap
        Heatmap colormap.
    vmax : float, optional
        Maximum heatmap color scale.
    ylim : tuple, optional
        y limits for total-output barplot.
    yticks : sequence, optional
        y ticks for total-output barplot.
    xlabel : str
        Heatmap x-axis label.

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = obs.copy()

    counts = (
        df.groupby([progenitor_col, fate_col], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=fate_order, fill_value=0)
    )

    total = counts.sum(axis=1)
    counts = counts.loc[total >= min_cells]
    total = total.loc[counts.index]

    frac = counts.div(counts.sum(axis=1), axis=0)

    if row_order == "cluster":
        g = sns.clustermap(
            frac,
            row_cluster=True,
            col_cluster=False,
        )
        plt.close(g.fig)

        order = g.dendrogram_row.reordered_ind
        frac_sorted = frac.iloc[order]

    elif row_order == "sort":
        sort_df = frac.copy()

        sort_cols = []

        if row_group_fn is not None:
            sort_df["_group"] = row_group_fn(sort_df)
            sort_cols.append("_group")

        if sort_col is not None:
            sort_cols.append(sort_col)

        if sort_cols:
            ascending = [True] * len(sort_cols)
            if sort_col is not None:
                ascending[-1] = sort_ascending

            sort_df = sort_df.sort_values(
                sort_cols,
                ascending=ascending,
            )

        frac_sorted = frac.loc[sort_df.index]

    else:
        raise ValueError("row_order must be 'cluster' or 'sort'")

    total = total.loc[frac_sorted.index]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=figsize,
        dpi=dpi,
        gridspec_kw={"height_ratios": [1, 3]},
        sharex=True,
    )

    sns.barplot(
        x=np.arange(len(total)),
        y=total.values,
        ax=axes[0],
        color="black",
        saturation=1,
        width=1,
    )

    sns.heatmap(
        frac_sorted[fate_order].T,
        cmap=cmap,
        yticklabels=True,
        ax=axes[1],
        cbar=False,
        vmax=vmax,
    )

    axes[1].set_xticks(np.arange(len(total)) + 0.5)
    axes[1].set_xticklabels("")
    axes[0].set_xticks([])

    if ylim is not None:
        axes[0].set_ylim(*ylim)

    if yticks is not None:
        axes[0].set_yticks(yticks)

    axes[0].set_ylabel("Total\noutput")
    axes[1].set_ylabel("")
    axes[1].set_xlabel(xlabel)

    fig.subplots_adjust(hspace=0)

    return fig


def plot_bipotent_probability_over_time(
    nc_progenitors,
    lineages=("Autonomic", "Sensory"),
    palette=None,
    xlim=(8, 9.5),
    ax=None,
):
    """Plot single-lineage, bipotent, and independent probabilities over time.

    Rows are binned by ``time`` into 0.25-wide bins (labeled by right edge)
    spanning ``xlim``. Per bin, plots the percentage of rows positive for each
    lineage, for both ("Bipotent"), and the product of the two single-lineage
    probabilities ("Independent", the expectation if fates were independent).

    Parameters
    ----------
    nc_progenitors : DataFrame
        One row per progenitor/clade, with a numeric ``time`` column and a
        boolean column for each lineage in ``lineages``.
    lineages : tuple of str
        The two lineage columns to compare.
    palette : dict, optional
        Colors for the two lineages, ``"Bipotent"``, and ``"Independent"``.
        A default palette is used if None.
    xlim : tuple of float
        X-axis limits, also used to define the time bins.
    ax : Axes, optional
        Axes to draw on. Creates a new 1.8 x 1.8 inch figure if None.

    Returns
    -------
    fig : Figure
        Figure containing the plot.
    """
    a, b = lineages
    df = nc_progenitors.copy()
    # set time bins based on xlim and a fixed bin width of 0.25
    time_bins = np.arange(xlim[0] - 0.25, xlim[1] + 0.25, 0.25)
    df["Bipotent"] = df[a] & df[b]
    df["time_bin"] = pd.cut(df["time"], bins=time_bins).apply(lambda x: x.right)
    df = df.dropna(subset=["time_bin"])

    probs = df.groupby("time_bin")[[a, b, "Bipotent"]].mean().reset_index()
    probs["Independent"] = probs[a] * probs[b]

    plot_df = probs.melt(
        id_vars="time_bin",
        value_vars=[a, b, "Bipotent", "Independent"],
        var_name="category",
        value_name="probability",
    ).sort_values("time_bin")
    plot_df["pct"] = 100 * plot_df["probability"]

    if palette is None:
        palette = {a: "#D55E00", b: "#0072B2", "Bipotent": "#FFAA00", "Independent": "#999999"}

    if ax is None:
        fig, ax = plt.subplots(figsize=(1.8, 1.8), dpi=600)
    else:
        fig = ax.figure

    sns.lineplot(
        data=plot_df,
        x="time_bin",
        y="pct",
        hue="category",
        palette=palette,
        ax=ax,
    )

    ax.set(
        xlim=xlim,
        xlabel="Time bin",
        ylabel="Clade fraction (%)",
    )
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    return fig
