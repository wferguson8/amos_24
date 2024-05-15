"""
Visualize the results of the election simulation
"""

import geopandas as gpd
import pandas as pd
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.pyplot as plt
import json

DATA = pd.read_json("./results_by_state.json", orient='index')
SF = "./shapefiles/States_shapefile-shp/States_shapefile.shp"
def create_gradient_cm(hex: str, steps: int=1, spread: int=25) -> ListedColormap:
    """
    Note: This code is AI Generated

    :param hex:
    :param steps:
    :return:
    """

    hex_color = hex.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    colors = [rgb]

    for i in range(1, steps + 1):
        step_color = [max((c - spread), 0) for c in rgb]
        colors.append(tuple(step_color))
    return ListedColormap(colors)

def single_candidate_outlook(candidate: str, start_color: str) -> None:
    """
    Visualize a single candidate's likelihood of winning an election

    :param candidate:
    :return:
    """

    # Create Map DF
    map_df = gpd.read_file(
        filename=SF
    )

    map_df = map_df[map_df['State_Code'] != "DC"]

    cmap = create_gradient_cm(start_color)

    combined = map_df.set_index("State_Code").join(DATA[candidate]).reset_index()

    fig, ax = plt.subplots(1, figsize=(10, 10))
    combined.plot(
        column=candidate,
        cmap=cmap,
        ax=ax
    )

    ax.axis("off")
    sm = plt.cm.ScalarMappable(cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax)

    plt.savefig(f"./{candidate}.png")


def lean_map() -> None:
    pass

if __name__ == "__main__":
    single_candidate_outlook("A", "#FF5733")