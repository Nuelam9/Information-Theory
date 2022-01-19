import numpy as np
import pandas as pd


def pmf_univariate(samples):
    n = len(samples)
    samples_list, pmf_vector = np.unique(samples, return_counts=True)
    return samples_list, pmf_vector / n


def joint_p(x: np.array, y: np.array) -> np.ndarray:
    """
    Compute the joint probability of x and y
    Args:
        x (np.array) : x feature
        y (np.array) : y feature
    Returns:
        np.ndarray: Joint Probability
    """
    return pd.crosstab(x, y, normalize=True).to_numpy()
