"""
Train Model using Classification Tree
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import json
from electoral_college import ec
from catboost import CatBoostClassifier

MODEL = CatBoostClassifier()  # No extra options for now

DATA = pd.DataFrame.from_dict(
    data=ec,
    orient='index',
    columns=["Electoral College"]
)

def train_model(x: np.ndarray, y: np.ndarray) -> None:
    """
    Train model using predefined data schema

    :param x: input data
    :param y: target data
    :return: None
    """

    features = x[:, 1:]

    cat_features = [4, 5, 6]

    x_train, x_test, y_train, y_test = train_test_split(
        features, y, test_size=0.1, random_state=10
    )

    MODEL.fit(x_train, y_train, cat_features=cat_features, eval_set=(x_test, y_test))

def predict(x: np.ndarray) -> np.ndarray:
    """
    Use the model to generate predictions

    :param x: The data you want to predict
    :return:  Predicted outcomes
    """

    pred = x[:, 1:]

    winners = MODEL.predict(pred)

    deltas = np.zeros(
        shape=(x.shape[0],  3)
    )

    deltas[:, 0] = DATA["Electoral College"]
    deltas[:, 1] = DATA["Electoral College"]
    deltas[:, 2] = DATA["Electoral College"]

    deltas[winners[:, 0] == "A", 1:3] = 0
    deltas[winners[:, 0] == "B", 0] = 0
    deltas[winners[:, 0] == "B", 2] = 0
    deltas[winners[:, 0] == "C", 0:2] = 0
    deltas[winners[:, 0] == "T", :] = 0 # For now: Drops all ties electoral votes

    return winners, deltas