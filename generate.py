"""
Generate Synthetic Data
"""

import numpy as np
from tqdm import tqdm
import json
from electoral_college import ec
import pandas as pd
from scipy.stats import alpha, binom
from typing import List

poll_data = pd.read_csv('./master_results.csv')
# TODO: Determine how to weight these values that are read in

def generate_state_pdf(state: str) -> List:
    """
    Generate the pdf distributions for a state

    :param row:
    :return:
    """

    state_data = poll_data[poll_data["State"] == state]
    vs_a = state_data["Candidate_A_VS"]
    vs_b = state_data["Candidate_B_VS"]
    vs_c = state_data["Candidate_C_VS"]

    vb_a = state_data["Party A"]
    vb_b = state_data["Party B"]
    vb_c = state_data["Party C"]

    ind = state_data["Independent"]
    abs = state_data["Absentee"]


    pdfs = []
    pdfs.append(alpha.fit(vs_a))
    pdfs.append(alpha.fit(vs_b))
    pdfs.append(alpha.fit(vs_c))
    pdfs.append(alpha.fit(vb_a))
    pdfs.append(alpha.fit(vb_b))
    pdfs.append(alpha.fit(vb_c))
    pdfs.append(alpha.fit(ind))
    pdfs.append(alpha.fit(abs))

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

if "__name__" == "__main__":
    compile()