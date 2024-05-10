"""
Contains Simulation Code
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from synthetics import synthetic_election
from electoral_college import ec
from train import train_model

DATA = pd.read_csv('./master_results.csv')
DATA.fillna(0, inplace=True)
DATA = DATA.to_numpy()

def main():
    """
    The main script of AMoS. Runs a Bunch of Elections simulations and outputs stats about the winner
    :return:
    """

    # TODO: Reformat this as command line args

    num_sims = 1

    # Train the model
    x = DATA[:, 0:-1]
    y = DATA[:, -1]

    train_model(x, y)

    # Run Simulations

    results = []
    elections_winners = {
        "A": 0,
        "B": 0,
        "C": 0,
    }

    for i in tqdm(range(num_sims), desc='Generating simulated elections'):
        e = synthetic_election()
        a_votes = e[:, 12].sum()
        b_votes = e[:, 13].sum()
        c_votes = e[:, 14].sum()

        winner = np.argmax([a_votes, b_votes, c_votes])

        if winner == 0:
            elections_winners["A"] += 1
        elif winner == 1:
            elections_winners["B"] += 1
        else:
            elections_winners["C"] += 1

        results.append(e)

    print("Elections successfully simulated.")

    results = np.concatenate(results, axis=1)  # Mega 2-D Array

    states = ec.keys()

    out = {

    }

    for state in tqdm(states, desc='Running State-By-State Analysis'):
        state_sims = results[results[:, 0] == state]
        out[state] = {}
        out[state]["A"] = state_sims[state_sims[:, 13] == "A"] / num_sims
        out[state]["B"] = state_sims[state_sims[:, 13] == "B"] / num_sims
        out[state]["C"] = state_sims[state_sims[:, 13] == "C"] / num_sims

    print("Writing results...")

    with open("results_by_state.json", "w") as outfile:
        json.dump(out, outfile)
        outfile.close()

    with open("summary.json", "w") as outfile:
        json.dump(elections_winners, outfile)
        outfile.close()

    print("Elections successfully simulated.")


if __name__ == "__main__":
    main()





