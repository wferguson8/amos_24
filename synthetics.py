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

modern_data = pd.read_csv("./master_results.csv")

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

def fill_row_with_modern_date(row: np.ndarray) -> np.ndarray:
    """
    A slight update to use modern simulations for amOS

    :param row:
    :return:
    """

    state = row[0]

    # If the state doesn't exist in the modern data, use historical data to compile estimates
    if state not in modern_data[modern_data.state == state].values:
        return fill_row(row)

    size = modern_data[modern_data.state == state]["size"].mean()

    poll_results_a = modern_data[modern_data["state"] == state]["vs_a"].mean()
    poll_results_b = modern_data[modern_data["state"] == state]["vs_b"].mean()
    poll_results_c = modern_data[modern_data["state"] == state]["vs_c"].mean()

    party_res_a = modern_data[modern_data.state == state]["party_a"].mean()
    party_res_b = modern_data[modern_data.state == state]["party_b"].mean()
    party_res_c = modern_data[modern_data.state == state]["party_c"].mean()
    party_res_i = modern_data[modern_data.state == state]["ind"].mean()
    party_res_abs = modern_data[modern_data.state == state]["abs"].mean()

    stds = [alpha[1] for alpha in  DATA[state]['alphas']]

    alphas = [
        [size, stds[0]],
        [poll_results_a, stds[1]],
        [poll_results_b, stds[2]],
        [poll_results_c, stds[3]],
        [party_res_a, stds[4]],
        [party_res_b, stds[5]],
        [party_res_c, stds[6]],
        [party_res_i, stds[7]],
        [party_res_abs, stds[8]]
    ]

    size = max([int(norm.rvs(*alphas[0])), 1])

    row[1] = size

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
    pers = max([norm.rvs(*alphas[4]), 0.01])
    vision = max([norm.rvs(*alphas[5]), 0.01])
    compassion = max([norm.rvs(*alphas[6]), 0.01])
    ind = max([norm.rvs(*alphas[7]), 0.01])
    abs = max([norm.rvs(*alphas[8]), 0.01])

    # Normalize vote shares
    voters = pers + vision + compassion + ind + abs

    row[8] = pers
    row[9] = vision
    row[10] = compassion
    row[11] = ind
    row[12] = abs

    return row


def fill_row(row: np.ndarray) -> None:
    """
    Sample data for a single state in a simulation

    :param row:
    :return:
    """
    state = row[0]

    alphas = DATA[state]['alphas']

    size = max([int(norm.rvs(*alphas[0])), 1])

    row[1] = size

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
    pers = max([norm.rvs(*alphas[4]), 0.01])
    vision = max([norm.rvs(*alphas[5]), 0.01])
    compassion = max([norm.rvs(*alphas[6]), 0.01])
    ind = max([norm.rvs(*alphas[7]), 0.01])
    abs = max([norm.rvs(*alphas[8]), 0.01])

    # Normalize vote shares
    voters = pers + vision + compassion + ind + abs

    row[8] = pers
    row[9] = vision
    row[10] = compassion
    row[11] = ind
    row[12] = abs

    return row
