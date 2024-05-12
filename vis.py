"""
Visualize the results of the election simulation
"""

import geopandas as gpd
import pandas as pd
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import json

# Read in global data resources
with open("./results_by_state.json", 'r') as file:
    raw_data = json.load(file)

DATA = pd.read_json(raw_data, orient='index')
SF = "./shapefiles/cb_2018_us_nation_20m.shp"
def create_gradient_cm(hex: str, steps: int=5) -> ListedColormap:
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
        step_color = [(c + (255 - c) * i / steps) / 255 for c in rgb]
        colors.append(step_color)
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

    cmap = create_gradient_cm(start_color)



def lean_map() -> None:
    pass