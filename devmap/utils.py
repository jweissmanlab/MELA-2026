import re
from pathlib import Path

import matplotlib.collections as mcoll
import matplotlib.image as mimage
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def convert_svg_text(svg_path, font="Arial"):
    """Convert the text elements in an svg file to font-family and font-size attributes"""
    with open(svg_path) as f:
        svg_content = f.readlines()
    with open(svg_path, "w") as file:
        for line in svg_content:
            if "<text" in line or "<tspan" in line:
                # Extract the style attribute
                style_match = re.search(r'style="([^"]*)"', line)
                if style_match:
                    style_content = style_match.group(1)
                    font_size_match = re.search(r"\s*(\d+)px", style_content)
                    font_family_match = re.search(r"\'([^\']+)\'", style_content)
                    font_size = font_size_match.group(1) if font_size_match else "10"
                    font_family = font_family_match.group(1) if font_family_match else font
                    font_family = font_family.replace("DejaVu Sans", font)
                    line = line.replace("style=", f'font-family="{font_family}" font-size="{font_size}" style=')
            file.write(line)


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
        convert_svg_text(path)
