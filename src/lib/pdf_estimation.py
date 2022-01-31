#!/usr/bin/env python3.9.5
import numpy as np
from typing import Tuple
from sklearn.neighbors import KernelDensity


def doane_formula(data: np.array) -> int:
    """
    Compute the optimal bin width with the Doane's formula:

    k = 1 + log_2(n) + log_2(n) * (1 + |g_1| / sigma_{g_1})
    where g_1 is the 3rd-moment-skewness and 
    sigma_{g_1} = sqrt{6 * (n - 2) / [(n + 1)(n + 3)]}

    Args:
        data (np.array): data whose bin is being sought

    Returns:
        int: optimal bin width
    """
    import scipy.stats as st
    import math
    
    N = len(data)
    skewness = st.skew(data)
    sigma_g1 = math.sqrt((6 * (N - 2)) / ((N + 1) * (N + 3)))
    num_bins = 1 + math.log(N, 2) + math.log(1 + abs(skewness) / sigma_g1, 2)
    return round(num_bins)


def get_binwidht(data: np.array, bw_method: str) -> Tuple[int, float]:
    """Estimate the optimal number of bins and the binwidht from the data,
       using one of the methods in literature.

    Args:
        data (np.array): data
        bw_method (str): methods to estimate the optimal 
                         number of bins. Defaults to 'doane'.

    Returns:
        Tuple[int, float]: optimal number of bins and binwidht
    """
    bin_edges = np.histogram_bin_edges(a=data, bins=bw_method)
    bins = bin_edges.size - 1
    h = np.diff(bin_edges)[0]
    return bins, h 


def kde_estimate(data: np.array, kernel: str, 
                 bw_method: str='doane') -> Tuple[np.ndarray, np.ndarray]:
    """Kernel Density Estimation

    Args:
        data (np.array): data
        kernel (str): kernel form
        bw_method (str, optional): methods to estimate the optimal 
                                   number of bins. Defaults to 'doane'.
                                   Possible choices are:
                                   - 'fd' (Freedman Diaconis Estimator);
                                   - doane';
                                   - 'scott';
                                   - 'stone';
                                   - 'rice';
                                   - 'sturges';
                                   - 'sqrt'.

    Returns:
        Tuple[np.ndarray, np.ndarray]: grid for plot, 
                                       kernel density estimation
    """
    x_grid = np.linspace(data.min(), data.max(), 5000)
    # Get the estimated binwidth
    bins, h = get_binwidht(data, bw_method=bw_method)
    # Kernel Density Estimation
    kde_skl = KernelDensity(bandwidth=h, kernel=kernel)
    # Fit the Kernel Density model on the data
    kde_skl.fit(data[:, np.newaxis])
    # Compute the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return x_grid, np.exp(log_pdf)
