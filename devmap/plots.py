"""Reusable plot functions for devmap figures."""

import matplotlib.pyplot as plt
import seaborn as sns

from .config import edit_palette

def barplot_with_points(
    data,
    x,
    y,
    hue=None,
    order=None,
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
        saturation=1,
        edgecolor="black",
        linewidth=0.8,
        legend=False,
        errorbar=errorbar,
        capsize=capsize,
        err_kws={"linewidth": 0.8},
        ax=ax,
    )
    sns.stripplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        order=order,
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
