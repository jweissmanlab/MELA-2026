from pathlib import Path

import matplotlib.collections as mcoll
import matplotlib.image as mimage
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import treedata as td
import pandas as pd

from .config import get_paths


def save_plot(path, fig=None, transparent=False, rasterize=False, dpi=600):
    """Save a plot as svg or png file"""
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

def load_data(data = "topology", scvi = False, characters = False):
    """Load the data"""
    base_path, _, _ = get_paths("data")
    data_path = base_path / "data"
    obs = pd.read_csv(data_path / "obs.csv", index_col=0, dtype={"clone": "str"})
    cell_types = pd.read_csv(data_path / "cell_types.csv", index_col=0)
    obs = obs.merge(cell_types[["cell_type","germ_layer","lineage","cluster"]], left_on = "cell_subtype", right_index=True)
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
        umap = pd.read_csv(data_path / "umap.csv", index_col=0)
        tdata.obsm["X_umap"] = umap.loc[tdata.obs_names].values
    else:
        raise ValueError("Invalid data value. Must be one of 'topology', 'counts', 'log1p', 'log1p_hvg', or 'umap'.")
    tdata.obs = obs.loc[tdata.obs_names].copy()
    if scvi:
        scvi = pd.read_csv(data_path / "scvi.csv", index_col=0)
        tdata.obsm["X_scvi"] = scvi.loc[tdata.obs_names].values
    if characters:
        characters = pd.read_csv(data_path / "characters.csv", index_col=0)
        tdata.obsm["characters"] = characters.reindex(tdata.obs_names, fill_value="-").values
    return tdata


