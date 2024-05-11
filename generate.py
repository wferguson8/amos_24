"""
Generate Synthetic Data
"""

import numpy as np
from tqdm import tqdm
import json
from electoral_college import ec
import pandas as pd
from scipy.stats import alpha, binom, norm
from typing import List

poll_data = pd.read_csv('./master_results_no_2019.csv')
poll_data.fillna(0, inplace=True)
poll_data = poll_data.to_numpy()


# TODO: Determine how to weight these values that are read in

def generate_state_pdf(state: str) -> List:
    """
    Generate the pdf distributions for a state

    :param row:
    :return:
    """

    state_data = poll_data[poll_data[:, 0] == state]
    size = state_data[:, 1].flatten().astype(int)

    vs_a = state_data[:, 2].flatten().astype(float)
    vs_b = state_data[:, 3].flatten().astype(float)
    vs_c = state_data[:, 4].flatten().astype(float)

    vb_a = state_data[:, 8].flatten().astype(float)
    vb_b = state_data[:, 9].flatten().astype(float)
    vb_c = state_data[:, 10].flatten().astype(float)

    ind = state_data[:, 11].flatten().astype(float)
    abs = state_data[:, 12].flatten().astype(float)

    pdfs = []
    pdfs.append(norm.fit(size))
    pdfs.append(norm.fit(vs_a))
    pdfs.append(norm.fit(vs_b))
    pdfs.append(norm.fit(vs_c))
    pdfs.append(norm.fit(vb_a))
    pdfs.append(norm.fit(vb_b))
    pdfs.append(norm.fit(vb_c))
    pdfs.append(norm.fit(ind))
    pdfs.append(norm.fit(abs))

    return pdfs


def apply_decay() -> None:
    """
    Weight the polls by recency (or any other relevant factors)

    :return:
    """
    pass


def compile() -> None:
    """

    Use to compile data from master results file

    :return:
    """

    # import csv
    states = ec.keys()

    output = {}

    for state in tqdm(states):
        output[state] = {
            "electoral_college": ec[state],
            "alphas": generate_state_pdf(state)
        }

    filepath = "./probability_distributions.json"
    with open(filepath, "w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    compile()
