import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from devmap.utils import save_plot
from devmap.config import set_theme, discrete_cmap
import matplotlib.colors as mcolors
import re

def sort_key(label):
    # e.g. "E8.5-R2-clone3" -> (8.5, 'R2', 'clone3')
    # parts = label.split("-")
    # num = float(re.search(r"[\d.]+", parts[0]).group())
    return float(label[1:4])#(num, *parts[1:])

df_early = pd.read_csv("integration/embryo_cross_early_sinkhorn_matrix.csv", index_col=0)
df_late = pd.read_csv("integration/embryo_cross_late_sinkhorn_matrix.csv", index_col=0)


for df, title in zip([df_early, df_late], ["early", "late"]):

    set_theme()

    index_labels = list(df.index)
    index_labels.sort(key=sort_key)

    column_labels = list(df.columns)
    column_labels.sort(key=sort_key)

    fig = plt.figure(figsize=(3,3), dpi=200)

    sns.heatmap(
        df.loc[index_labels, column_labels].astype(float),
        xticklabels=column_labels,
        yticklabels=index_labels,
        cmap="viridis",
        cbar_kws={'label': 'Sinkhorn distance'},
        square=True,

    )

    save_plot(f"integration/embryo_cross_{title}_sinkhorn_heatmap.svg", fig)


