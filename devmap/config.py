from pathlib import Path
from importlib import resources

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Paths
base_path = Path(__file__).resolve().parent.parent

# Default paths
def get_paths(folder):
    """Get the paths for the data and plots folders"""
    folder = Path(folder)
    if "/" in str(folder):
        base_path = folder.parent
        folder = folder.name
    else:
        base_path = Path(__file__).resolve().parent.parent
    return base_path, base_path / folder / "plots", base_path / folder / "results"

# Default colors
colors = [
    "black",
    "#1874CD",
    "#CD2626",
    "#FFE600",
    "#009E73",
    "#8E0496",
    "#E69F00",
    "#83A4FF",
    "#DB65D2",
    "#75F6FC",
    "#7BE561",
    "#FF7D7D",
    "#7C0EDD",
    "#262C6B",
    "#D34818",
    "#20C4AC",
    "#A983F2",
    "#FAC0FF",
    "#7F0303",
    "#845C44",
    "#343434",
]

# Sequential colormaps
sequential_colors = [(1, 1, 1)] + [plt.cm.GnBu(i / (256 - 1)) for i in range(256)]
sequential_cmap = mcolors.LinearSegmentedColormap.from_list("GnBu", sequential_colors, N=256)
cooc_cmap = mcolors.LinearSegmentedColormap.from_list(
     "custom_diverging",
     ["#149A9A", "#f7f7f7","#B35797" ]  # low → midpoint → high
)

# Discrete palettes
discrete_colors = {
    1: ["black"],
    2: ["#CD2626", "#1874CD"],
    3: ["#CD2626", "#FFE600", "#1874CD"],
    4: ["#CD2626", "#FFE600", "#009E73", "#1874CD"],
    5: ["#CD2626", "#FFE600", "#009E73", "#1874CD", "#8E0496"],
    6: ["#CD2626", "#E69F00", "#FFE600", "#009E73", "#1874CD", "#8E0496"],
    7: ["#CD2626", "#E69F00", "#FFE600", "#009E73", "#83A4FF", "#1874CD", "#8E0496"],
    8: ["#CD2626", "#E69F00", "#FFE600", "#009E73", "#83A4FF", "#1874CD", "#8E0496", "#DB65D2"],
    9: ["#CD2626", "#E69F00", "#FFE600", "#009E73", "#75F6FC", "#83A4FF", "#1874CD", "#8E0496", "#DB65D2"],
    10: ["#CD2626", "#E69F00", "#FFE600", "#7BE561", "#009E73", "#75F6FC", "#83A4FF", "#1874CD", "#8E0496", "#DB65D2"],
    11: [
        "#FF7D7D",
        "#CD2626",
        "#E69F00",
        "#FFE600",
        "#7BE561",
        "#009E73",
        "#75F6FC",
        "#83A4FF",
        "#1874CD",
        "#8E0496",
        "#DB65D2",
    ],
    12: [
        "#FF7D7D",
        "#CD2626",
        "#E69F00",
        "#FFE600",
        "#7BE561",
        "#009E73",
        "#75F6FC",
        "#83A4FF",
        "#1874CD",
        "#7C0EDD",
        "#8E0496",
        "#DB65D2",
    ],
    13: [
        "#FF7D7D",
        "#CD2626",
        "#E69F00",
        "#FFE600",
        "#7BE561",
        "#009E73",
        "#75F6FC",
        "#262C6B",
        "#83A4FF",
        "#1874CD",
        "#7C0EDD",
        "#8E0496",
        "#DB65D2",
    ],
    14: [
        "#FF7D7D",
        "#CD2626",
        "#D34818",
        "#E69F00",
        "#FFE600",
        "#7BE561",
        "#009E73",
        "#75F6FC",
        "#262C6B",
        "#83A4FF",
        "#1874CD",
        "#7C0EDD",
        "#8E0496",
        "#DB65D2",
    ],
    15: [
        "#FF7D7D",
        "#CD2626",
        "#D34818",
        "#E69F00",
        "#FFE600",
        "#7BE561",
        "#009E73",
        "#20C4AC",
        "#75F6FC",
        "#262C6B",
        "#83A4FF",
        "#1874CD",
        "#7C0EDD",
        "#8E0496",
        "#DB65D2",
    ],
    16: [
        "#FF7D7D",
        "#CD2626",
        "#D34818",
        "#E69F00",
        "#FFE600",
        "#7BE561",
        "#009E73",
        "#20C4AC",
        "#75F6FC",
        "#262C6B",
        "#83A4FF",
        "#1874CD",
        "#A983F2",
        "#7C0EDD",
        "#8E0496",
        "#DB65D2",
    ],
    17: [
        "#FF7D7D",
        "#CD2626",
        "#D34818",
        "#E69F00",
        "#FFE600",
        "#7BE561",
        "#009E73",
        "#20C4AC",
        "#75F6FC",
        "#262C6B",
        "#83A4FF",
        "#1874CD",
        "#A983F2",
        "#7C0EDD",
        "#8E0496",
        "#DB65D2",
        "#FAC0FF",
    ],
    18: [
        "#FF7D7D",
        "#CD2626",
        "#7F0303",
        "#D34818",
        "#E69F00",
        "#FFE600",
        "#7BE561",
        "#009E73",
        "#20C4AC",
        "#75F6FC",
        "#262C6B",
        "#83A4FF",
        "#1874CD",
        "#A983F2",
        "#7C0EDD",
        "#8E0496",
        "#DB65D2",
        "#FAC0FF",
    ],
    19: [
        "#FF7D7D",
        "#CD2626",
        "#7F0303",
        "#D34818",
        "#E69F00",
        "#845C44",
        "#FFE600",
        "#7BE561",
        "#009E73",
        "#20C4AC",
        "#75F6FC",
        "#262C6B",
        "#83A4FF",
        "#1874CD",
        "#A983F2",
        "#7C0EDD",
        "#8E0496",
        "#DB65D2",
        "#FAC0FF",
    ],
    20: [
        "#FF7D7D",
        "#CD2626",
        "#7F0303",
        "#D34818",
        "#E69F00",
        "#845C44",
        "#FFE600",
        "#7BE561",
        "#009E73",
        "#20C4AC",
        "#75F6FC",
        "#262C6B",
        "#83A4FF",
        "#1874CD",
        "#A983F2",
        "#7C0EDD",
        "#8E0496",
        "#DB65D2",
        "#FAC0FF",
        "#343434",
    ],
}
discrete_cmap = {k: sns.color_palette(v) for k, v in discrete_colors.items()}

# Custom palettes
germ_layer_palette = {
    "Ectoderm": "#1874CD",
    "Mesoderm": "#CD2626",
    "Endoderm": "#FFE600",
    "Epiblast": "#A983F2",
    "Primordial germ cell": "#20C4AC",
}

stage_palette = {
 'E7.5': '#8000ff',
 'E8.0': '#1996f3',
 'E8.5': '#4df3ce',
 'E9.0': '#b2f396',
 'E9.5': '#ff964f',
 'E10.0': '#ff0000'}

type_palette = {
  "donor":"#1874CD",
  "host":"#FF7D7D"}

phase_palette = {
    'G0/G1': "#7C0EDD",
    'G2/M': "#20C4AC",
    'S': "#FFF600",
}

lineage_palette = {'Epiblast': "#A983F2",
                   'Primordial germ cell': "#20C4AC",
                   'Ectoderm': "#1874CD",
                   'Neural ectoderm': "#83A4FF", 
                   'Surface ectoderm': "#75F6FC", 
                   'Neural crest': "#7C0EDD", 
                   'Extraembryonic ectoderm':"#262C6B",
                   'Mesoderm': "#CD2626",
                   'Lateral plate mesoderm':"#7F0303",
                   'Intermediate mesoderm': "#FFC0CB",
                   'Paraxial mesoderm': "#FF7D7D", 
                   'Extraembryonic mesoderm': "#D34818", 
                   'Endoderm': "#FFE600",
                   'Extraembryonic endoderm': "#E69F00",
                   'Blood': "#009E73"}


embryos = ["E7.5-R1","E7.5-R2","E7.5-R3","E8.0-R1","E8.0-R2","E8.0-R3","E8.5-R1",
"E8.5-R2","E8.5-R3","E9.0-R1","E9.0-R2","E9.0-R3","E9.5-R1","E9.5-R2","E9.5-R3","E10.0-R1"]
embryo_palette = dict(zip(embryos,reversed(discrete_cmap[17][1:17])))

edit_palette = {str(i): discrete_cmap[8][i - 1] for i in range(1, 9)}
edit_palette.update({ "!": "#505050","*":"lightgray","-": "white"})

# Cell type palette
cell_types = pd.read_csv(base_path / "data" / "cell_types.csv", index_col=0)
subtype_palette = cell_types["subtype_color"].to_dict()
celltype_palette = cell_types.groupby("cell_type")["type_color"].first().to_dict()

# Default style
def set_theme(figsize=(3, 3), dpi=200):
    """Set the default style for the plots"""
    style_path = resources.files("devmap").joinpath("style.yaml")
    plt.style.use(str(style_path))
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colors[1:])
