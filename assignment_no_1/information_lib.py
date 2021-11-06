import numpy as np


def entropy(px: np.array) -> float:
    return - np.nansum(px * np.log2(px))


def joint_entropy(pxy: np.ndarray) -> float:
    return - np.nansum(pxy * np.log2(pxy))


def conditional_entropy(pxy: np.ndarray, px: np.array) -> float:
    return joint_entropy(pxy) - entropy(px)


def mutual_information(pxy: np.ndarray, px: np.array, py: np.array) -> float:
    return entropy(px) + entropy(py) - joint_entropy(pxy)


def norm_cond_entropy(pxy: np.ndarray, px: np.array) -> float:
    return conditional_entropy(pxy, px) / entropy(px)


def norm_joint_entropy(pxy: np.ndarray, px: np.array, py: np.array) -> float:
    return 1. - mutual_information(pxy, px, py) / (entropy(px) + entropy(py))


def norm_mutual_information1(pxy: np.ndarray, px: np.array, py: np.array) -> float:
    return 1. / norm_joint_entropy(pxy, px, py) - 1.


def norm_mutual_information2(pxy: np.ndarray, px: np.array, py: np.array) -> float:
    return 1. + norm_mutual_information1(pxy, px, py)


def norm_mutual_information3(pxy: np.ndarray, px: np.array, py: np.array) -> float:
    return mutual_information(pxy, px, py) / np.sqrt(entropy(px) * entropy(py))
