"""
Train Model using Classification Tree
"""

from sklearn.tree import DecisionTreeClassifier
import numpy as np

MODEL = DecisionTreeClassifier()
def train_model(x: np.ndarray, y: np.ndarray) -> None:
    """
    Train model using predefined data schema

    :param x: input data
    :param y: target data
    :return: None
    """

    MODEL.fit(x, y)

def predict(x: np.ndarray) -> np.ndarray:
    """
    Use the model to generate predictions

    :param x: The data you want to predict
    :return:  Predicted outcomes
    """

    return MODEL.predict(x)