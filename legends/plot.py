import matplotlib.pyplot as plt
import sys
from pathlib import Path
import matplotlib as mpl
import numpy as np

plots_path = Path(__file__).parent / "plots"
base_path = Path(__file__).parent.parent
sys.path.append(str(base_path))
plt.style.use(base_path / 'plot.mplstyle')

from devmap.utils import save_plot
from devmap.config import sequential_cmap

def add_cbar(cbar_ax, cmap, ticks, label, ticklabels=None, center=None):
    vmin = min(ticks)
    vmax = max(ticks)
    if center is not None:
        norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
    else:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = mpl.colorbar.ColorbarBase(
        cbar_ax, cmap=cmap, norm=norm, orientation='vertical'
    )
    cbar.outline.set_visible(False)
    cbar.set_label(label, labelpad=3)
    cbar.set_ticks(ticks)
    if ticklabels is not None:
        cbar.ax.set_yticklabels(ticklabels)
    else:
        cbar.set_ticklabels([f'{x}' for x in ticks])
    
    cbar.ax.yaxis.set_tick_params(pad=2)

def plot_legend(handels, name, title= None):
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.plot([], [])
    fig.legend(handles=handels,loc='upper right',bbox_to_anchor=(.995,1),ncol=1,title=title)
    ax.axis('off')
    save_plot(fig, name, plots_path)

def plot_size_legend(name, pval_ticks=None, min_size=2, max_size=10):
    if pval_ticks is None:
        pval_ticks = [0.01, 0.05, 0.1, 0.5, 1.0]

    s_min, s_max = min_size, max_size
    # Reproduce the same raw_sizes range from your data
    data_min, data_max = 0.01, 1.0
    raw_min = -np.log10(data_max)   # -log10(1) = 0
    raw_max = -np.log10(data_min)   # -log10(0.01) = 2
    size_range = raw_max - raw_min

    legend_raw = -np.log10(np.array(pval_ticks))
    legend_sizes = s_min + (legend_raw - raw_min) / size_range * (s_max - s_min)

    fig, ax = plt.subplots(figsize=(1, 1.5))
    for i, (pval, sz) in enumerate(zip(pval_ticks, legend_sizes)):
        ax.scatter(0.5, i, s=sz, c="white", edgecolors="black", linewidths=0.3)
        ax.text(0.7, i, f"{pval}", va="center", fontsize=6)

    ax.set_xlim(0, 1.5)
    ax.set_ylim(-0.5, len(pval_ticks) - 0.5)
    ax.set_ylabel("P-value", fontsize=7)
    ax.axis("off")
    save_plot(plots_path / f"{name}.svg", fig)

def plot_cbar(cmap, ticks, label, name, ticklabels=None,center=None):
    fig, cbar_ax = plt.subplots(figsize=(.2, 2))
    add_cbar(cbar_ax, cmap, ticks, label, ticklabels=ticklabels,center=center)
    save_plot(plots_path / f"{name}.svg", fig)

if __name__ == "__main__":
    plot_cbar(mpl.colormaps["RdBu_r"], [-1.5,0,1.5], "Mean Linkage", "mean_linkage_cbar")
    plot_cbar(mpl.colormaps["RdBu_r"], [-2,0,3], "Epiblast Linkage", "epiblast_linkage_cbar", center = 0)
    plot_cbar(mpl.colormaps["Grays"], [0, 0.1, 0.2], "Linkage Variance", "linkage_variance_cbar")
    plot_cbar(sequential_cmap, [0, 0.5, 1], "Clade Fraction", "clade_fraction_cbar")
    plot_size_legend("linkage_pval_legend", min_size=2, max_size=10)
    plot_size_legend("germ_layer_linkage_pval_legend", min_size=2, max_size=20)