"""
Samples a synthetic election
"""

import numpy as np
import pandas as pd
from scipy.stats import alpha
from train import predict
from electoral_college import ec, home_states

DATA = pd.read_csv("./master_results.csv")
DATA.fillna(0, inplace=True)
DATA = DATA.to_numpy()
def synthetic_election() -> np.ndarray:

    # columns:
    # State, Size, A_VS, B_VS, C_VS,  A_HS, B_HS, C_HS, PA, PB, PC, PI, AB, [Winner], EC_A, EC_B, EC_C

    states = ec.keys()

    election = np.zeros(shape=(
        len(states), 16
    ))

    states = np.array(states).reshape((1, len(states))) # Transpose states

    election[:, 0] = states

    # generate - Row 12 should still be empty
    election = np.apply_along_axis(
        fill_row,
        1,
        election
    )

    winners = predict(election[:, 0:12])
    election[:, 12] = np.array(winners).reshape((1, len(states))) # Fill last column with winner of each state

    # Update electoral college columns
    election[:, 13] = DATA[election[:, 0]]["electoral_college"] if election[:, 11] == "A" else 0
    election[:, 14] = DATA[election[:, 0]]["electoral_college"] if election[:, 11] == "B" else 0
    election[:, 15] = DATA[election[:, 0]]["electoral_college"] if election[:, 11] == "C" else 0

    return election

def fill_row(row: np.ndarray) -> np.ndarray:
    """
    Sample data for a single state in a simulation

    :param row:
    :param beta:
    :return:
    """
    state = row[0]

    alphas = DATA[state]['alphas']

    row[1] = alpha.rvs(*alphas[0])

    # Candidate Vote Share Sample (rows 1-3)
    candidate_a = alpha.rvs(*alphas[1])
    candidate_b = alpha.rvs(*alphas[2])
    candidate_c = alpha.rvs(*alphas[3])
    total = candidate_a + candidate_b + candidate_c

    a_vs = candidate_a / total * 100
    b_vs = candidate_b / total * 100
    c_vs = candidate_c / total * 100

    row[2] = a_vs
    row[3] = b_vs
    row[4] = c_vs

    # Home State Categorization (rows 4-6)
    row[5] = 1 if state in home_states["A"] else 0
    row[6] = 1 if state in home_states["B"] else 0
    row[7] = 1 if state in home_states["C"] else 0

    # Vote Shares (rows 7-11)
    row[8] = alpha.rvs(*alphas[4])
    row[9] = alpha.rvs(*alphas[5])
    row[10] = alpha.rvs(*alphas[6])
    row[11] = alpha.rvs(*alphas[7])
    row[12] = alpha.rvs(*alphas[8])

