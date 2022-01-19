import math
import numpy as np
import scipy.stats as st
from sklearn.neighbors import KernelDensity
from typing import Tuple


def doane_formula(data: np.array) -> float:
    """
    Compute the optimal bin width with the Doane's formula:
    k = 1 + \log_{2}(n) + \log_{2}\left (1 + \frac{\left |g_{1}  
    \right |}{\sigma_{g_{1}}}  \right )
    where g_{1} is the 3rd-moment-skewness and 
    \sigma_{g_{1}} = \sqrt{\frac{6(n - 2)}{(n + 1)(n + 3)}}

    Args:
        data (np.array): data whose bin is being sought

    Returns:
        float: optimal bin width
    """
    N = len(data)
    skewness = st.skew(data)
    sigma_g1 = math.sqrt((6 * (N - 2)) / ((N + 1) * (N + 3)))
    num_bins = 1 + math.log(N, 2) + math.log(1 + abs(skewness) / sigma_g1, 2)
    num_bins = round(num_bins)
    return num_bins


def kde_estimate(x: np.array, kernel: str) -> Tuple[np.ndarray, np.ndarray]:
    """Kernel Density Estimation

    Args:
        x (np.array): data
        kernel (str): kernel form

    Returns:
        tuple(np.ndarray, np.array): grid for plot, kernel density estimation
    """
    x_grid = np.linspace(x.min(), x.max(), 5000)
    h  = doane_formula(x)
    kde_skl = KernelDensity(bandwidth=h, kernel=kernel)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return x_grid, np.exp(log_pdf)
