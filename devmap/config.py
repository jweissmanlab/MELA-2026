from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
base_path = Path(__file__).resolve().parent.parent

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

# Sequential colormap
sequential_colors = [(1, 1, 1)] + [plt.cm.GnBu(i / (256 - 1)) for i in range(256)]
sequential_cmap = mcolors.LinearSegmentedColormap.from_list("GnBu", sequential_colors, N=256)

# Discrete colormap
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

# Edit colors
edit_palette = {str(i): discrete_cmap[8][i - 1] for i in range(1, 9)}
edit_palette.update({ "!": "#505050","*":"lightgray","-": "white"})

# Default style
def set_theme(figsize=(3, 3), dpi=200):
    """Set the default style for the plots"""
    plt.style.use(base_path / "plot.mplstyle")
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colors[1:])

# Default paths
def get_paths(folder):
    """Get the paths for the data and plots folders"""
    folder = Path(folder)
    if "/" in str(folder):
        base_path = folder.parent
        folder = folder.name
    else:
        base_path = Path(__file__).resolve().parent.parent
    return base_path, base_path / folder / "data", base_path / folder / "plots"
