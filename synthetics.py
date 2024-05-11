"""
Samples a synthetic election
"""

import numpy as np
import pandas as pd
import json
from scipy.stats import alpha, norm
from train import predict
from electoral_college import ec, home_states

with open("./probability_distributions.json") as file:
    DATA = json.load(file)

def synthetic_election() -> np.ndarray:
    # columns:
    # State, Size, A_VS, B_VS, C_VS,  A_HS, B_HS, C_HS, PA, PB, PC, PI, AB, [Winner], EC_A, EC_B, EC_C

    states = list(ec.keys())

    election = np.zeros(
        shape=(len(states), 17),
        dtype=object
    )

    states = np.array(states)

    election[:, 0] = states

    # generate - Row 12 should still be empty
    election = np.apply_along_axis(
        fill_row,
        1,
        election
    )

    winners, deltas = predict(election[:, 0:13])
    election[:, 13] = np.array(winners).reshape((1, len(states)))  # Fill last column with winner of each state

    # Update electoral college columns
    election[:, 14:] = deltas

    return election


def fill_row(row: np.ndarray) -> None:
    """
    Sample data for a single state in a simulation

    :param row:
    :return:
    """
    state = row[0]

    alphas = DATA[state]['alphas']

    row[1] = max([int(norm.rvs(*alphas[0])), 1])

    # Candidate Vote Share Sample (rows 1-3)
    candidate_a = max([norm.rvs(*alphas[1]), 0.01]) # Enforces no negative values
    candidate_b = max([norm.rvs(*alphas[2]), 0.01])
    candidate_c = max([norm.rvs(*alphas[3]), 0.01])
    total = candidate_a + candidate_b + candidate_c

    a_vs = (candidate_a / total)
    b_vs = (candidate_b / total)
    c_vs = (candidate_c / total)

    row[2] = a_vs
    row[3] = b_vs
    row[4] = c_vs

    # Home State Categorization (rows 4-6)
    row[5] = 1 if state in home_states["A"] else 0
    row[6] = 1 if state in home_states["B"] else 0
    row[7] = 1 if state in home_states["C"] else 0

    # Vote Shares (rows 7-11)
    pers = max([norm.rvs(*alphas[4]), 0])
    vision = max([norm.rvs(*alphas[5]), 0])
    compassion = max([norm.rvs(*alphas[6]), 0])
    ind = max([norm.rvs(*alphas[7]), 0])
    abs = max([norm.rvs(*alphas[8]), 0])

    # Normalize vote shares
    voters = pers + vision + compassion + ind + abs

    row[8] = (pers / voters)
    row[9] = (vision / voters)
    row[10] = (compassion / voters)
    row[11] = (ind / voters)
    row[12] = (abs / voters)

    return row
