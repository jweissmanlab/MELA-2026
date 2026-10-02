from pathlib import Path

import matplotlib.collections as mcoll
import matplotlib.image as mimage
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import treedata as td

from .config import get_paths


def save_plot(path, fig=None, transparent=False, rasterize=False, dpi=600):
    """Save a figure as an SVG or PNG file.

    The output format is chosen from the file suffix; figures are saved with
    ``bbox_inches="tight"`` and no padding. Paths with any other suffix are
    silently ignored (nothing is written).

    Parameters
    ----------
    path : str or Path
        Output file path ending in ``.svg`` or ``.png``.
    fig : Figure, optional
        Figure to save. Defaults to the current figure (``plt.gcf()``).
    transparent : bool
        Whether to save with a transparent background.
    rasterize : bool
        For SVG output only, rasterize dense artists (path/poly/patch/line
        collections, quad meshes, patches, and images) on every axes, keeping
        text and axes as vector elements.
    dpi : int
        Resolution for PNG output and rasterized SVG elements.

    Returns
    -------
    None
    """
    if fig is None:
        fig = plt.gcf()
    suffix = Path(path).suffix
    if suffix == ".png":
        fig.savefig(path, bbox_inches="tight", pad_inches=0, transparent=transparent, dpi=dpi)
    elif suffix == ".svg":
        if rasterize:
            for ax in fig.axes:
                for artist in ax.get_children():
                    if isinstance(
                        artist,
                        mcoll.PathCollection
                        | mcoll.PolyCollection
                        | mcoll.QuadMesh
                        | mcoll.PatchCollection
                        | mcoll.LineCollection
                        | mpatches.Patch
                        | mpatches.Rectangle
                        | mimage.AxesImage,
                    ):
                        artist.set_rasterized(True)
        fig.savefig(path, bbox_inches="tight", pad_inches=0, transparent=transparent, dpi=dpi)


def load_data(data="topology", scvi=False, characters=False):
    """Load the project TreeData with merged metadata and embeddings.

    Reads ``data/obs.csv`` (with ``clone`` as string) and inner-joins it with the
    ``cell_type``, ``germ_layer``, ``lineage``, and ``cluster`` columns of
    ``data/cell_types.csv`` on ``cell_subtype``. The selected TreeData is then
    given this merged ``obs`` (aligned to its ``obs_names``) and the UMAP
    coordinates from ``data/umap.csv``.

    Parameters
    ----------
    data : str
        Which dataset to load:

        - ``"topology"``: ``data/topology.h5td`` (lineage trees in ``obst``).
        - ``"counts"``: ``data/counts.h5td`` (raw counts).
        - ``"log1p"``: ``data/log1p.h5td`` (log1p-normalized expression, all genes).
        - ``"log1p_hvg"``: ``data/log1p_hvg.h5td`` (log1p expression, highly variable genes).
        - ``"umap"``: no ``.h5td`` file is read; an obs-only ``TreeData`` is built
          from the merged ``obs`` (no expression or trees).
    scvi : bool
        If True, add the scVI latent space from ``data/scvi.csv`` as
        ``obsm["X_scvi"]``.
    characters : bool
        If True, add the allele/character table from ``data/characters.csv`` as
        ``obsm["characters"]`` (a DataFrame reindexed to ``obs_names``; cells
        missing from the file are filled with ``"-"``).

    Returns
    -------
    tdata : TreeData
        TreeData with merged ``obs``, ``obsm["X_umap"]``, and optionally
        ``obsm["X_scvi"]`` and ``obsm["characters"]``.

    Raises
    ------
    ValueError
        If ``data`` is not one of the options listed above.
    """
    base_path, _, _ = get_paths("data")
    data_path = base_path / "data"
    obs = pd.read_csv(data_path / "obs.csv", index_col=0, dtype={"clone": "str"})
    cell_types = pd.read_csv(data_path / "cell_types.csv", index_col=0)
    obs = obs.merge(
        cell_types[["cell_type", "germ_layer", "lineage", "cluster"]], left_on="cell_subtype", right_index=True
    )
    if data == "topology":
        tdata = td.read_h5td(data_path / "topology.h5td")
    elif data == "counts":
        tdata = td.read_h5td(data_path / "counts.h5td")
    elif data == "log1p":
        tdata = td.read_h5td(data_path / "log1p.h5td")
    elif data == "log1p_hvg":
        tdata = td.read_h5td(data_path / "log1p_hvg.h5td")
    elif data == "umap":
        tdata = td.TreeData(obs=obs)
    else:
        raise ValueError("Invalid data value. Must be one of 'topology', 'counts', 'log1p', 'log1p_hvg', or 'umap'.")
    tdata.obs = obs.loc[tdata.obs_names].copy()
    umap = pd.read_csv(data_path / "umap.csv", index_col=0)
    tdata.obsm["X_umap"] = umap.loc[tdata.obs_names].values
    if scvi:
        scvi = pd.read_csv(data_path / "scvi.csv", index_col=0)
        tdata.obsm["X_scvi"] = scvi.loc[tdata.obs_names].values
    if characters:
        characters = pd.read_csv(data_path / "characters.csv", index_col=0)
        tdata.obsm["characters"] = characters.reindex(tdata.obs_names, fill_value="-")
    return tdata
