#!/usr/bin/env python3.9.5
import os
import numpy as np
from lib.utils import *
from lib.analysis_plot import *
from lib.information_lib import Entropy


if __name__ == '__main__':
    # Number of samples to generate for p0 (M)
    M = 100
    p0 = np.linspace(0, 1, M)
    p = np.vstack([p0, 1 - p0]).T

    # Computing the Entropy of a generic binary random variable
    # as a function of p.m.f. p
    entropy = list(map(Entropy, p))

    file = os.path.basename(__file__).split('.')[0]
    # Save probability and entropy vectors
    np.save(f'../Results/{file}/entropy_vs_p0.npy', np.vstack((p0, entropy)))
