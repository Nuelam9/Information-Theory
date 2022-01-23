import numpy as np


def random_data_gen(x: float, N: int) -> np.array:
    """Generate random numbers between inf and sup with a given 
       probability x.

    Args:
        x (float): probability
        N (int): desired size of random data

    Return:
        np.array: random numbers between inf and sup
    """
    return np.random.choice(a=[0, 1], size=N, p=[x, 1 - x])


def prob(x: np.array) -> np.array:
    """Compute the probability of each element of x.

    Args:
        x (np.array): random numbers vector

    Return:
        np.array: probability of each element of x
    """
    n = len(x)
    _, freq = np.unique(x, return_counts=True)
    return freq / n


def Gaussian(x: np.array, muj: float, varj: float) -> np.array:
    """Compute the the Gaussian density function of a vector x

    Args:
        x (np.array): features vector
        muj (float): mean
        varj (float): variance

    Returns:
        np.array: [description]
    """   
    pdf = np.sqrt(2. * np.pi * varj) * np.exp(- 0.5 * (x - muj) ** 2. / varj)
    return pdf
