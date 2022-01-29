#!/usr/bin/env python3.9.5
from lib.information_lib import Entropy, Diff_entropy
from multiprocessing import Pool, cpu_count
from lib.pdf_estimation import kde_estimate
from lib.utils import random_data_gen, prob
from lib.analysis_plot import *
from itertools import product
import pandas as pd
import numpy as np
import os


def sampling(n: int, bw_method: str, kernel: str) -> float:
    """Get the differential entropy of random variable vector sampled
       from a gaussian distribution with mean = 0 and variance = 1.

    Args:
        n (int): samples size
        bw_method (str): binwidht method
        kernel (str): kernel function 

    Returns:
        [float]: differential entropy with a given n, binwidht method
                 and kernel function 
    """
    # Samples generated through a gaussian p.d.f.
    # with mean = 0, variance = 1
    samples = np.random.normal(loc=0.0, scale=1.0, size=n)
    # Apply the kernel density estimation
    x, est_pdf = kde_estimate(samples, kernel, bw_method)
    return Diff_entropy(est_pdf, x)


if __name__ == '__main__': 
    # Set the number of samples to generate for p0 (M)
    M = 100
    p0 = np.linspace(0, 1, M)
    p = np.vstack([p0, 1 - p0]).T

    # Computing the Entropy of the truth p.m.f.
    entropy = list(map(Entropy, p))

    # Define the N samples size array
    n_samples = 10 ** np.arange(2, 6)

    est_entropies = []
    # Cycle over the size of random data (N)
    for N in n_samples:
        # Computing the entropy for the p0 values contained in p array
        x = [random_data_gen(p, N) for p in p0]
        # Compute the estimated pmf
        pk = list(map(prob, x))
        # Compute the entropy from the estimated p.m.f.
        est_entropy = list(map(Entropy, pk))
        est_entropies.append(est_entropy)

    # Saving the results for the entropies as .csv file
    n = len(n_samples)
    res = np.zeros((M*n, 2))

    for i, N in enumerate(n_samples):
        res[M*i:M*(i+1), :] = np.column_stack(([N]*M, est_entropies[i]))
        
    df = pd.DataFrame(res, columns=['n_samples', 'est_entropy'])

    tmp = [np.subtract(entropy, h_est) for h_est in est_entropies]
    difference = [item for sublist in tmp for item in sublist]

    df['difference'] = difference
    file = os.path.basename(__file__).split('.')[0]
    df.to_csv(f'../Results/{file}/samples_entropy_1.csv', index=False)    

   ########################Differential entropy#########################

    # Define the kernel list to use 
    kernels = ['gaussian', 'tophat', 'epanechnikov',
                'exponential', 'linear', 'cosine']
    
    # Define the estimation methods to use for the optimal number of bins
    bw_methods = ['fd', 'doane', 'scott', 'stone', 'rice', 'sturges', 'sqrt']

    # Define the N samples size array
    n_samples = 10 ** np.arange(2, 6)    
    
    # Define the number of worker processes to use
    processes = cpu_count() - 2
    with Pool(processes=processes) as pool:
        results = pool.starmap(sampling,
                               product(n_samples, bw_methods, kernels))

    # Save the results as .csv file
    df = pd.DataFrame(product(n_samples, bw_methods, kernels),
                      columns=['n_samples', 'bw_method', 'kernel'])
    
    df['results'] = results
    df.to_csv('../Results/assignment_no_2/diff_entropy_2.csv', index=False)
    