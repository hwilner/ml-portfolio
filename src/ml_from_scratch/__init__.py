"""NumPy-only implementations of core machine-learning algorithms.

Written from scratch so that the update rules, split criteria and optimisation
loops are visible rather than hidden behind a library. Each module pairs a
fully documented implementation with the exploratory notebook it was extracted
from.
"""

from .decision_tree import DecisionTreeClassifier, information_gain, shannon_entropy
from .linear_regression import LinearRegression, MinMaxScaler
from .perceptron import Perceptron

__version__ = "1.0.0"

__all__ = [
    "DecisionTreeClassifier",
    "LinearRegression",
    "MinMaxScaler",
    "Perceptron",
    "information_gain",
    "shannon_entropy",
]
