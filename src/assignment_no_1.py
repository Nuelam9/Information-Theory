#!/usr/bin/env python3.9.5
import os
from lib.utils import *
import numpy as np
import matplotlib.pyplot as plt
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
    
    plot_settings() # for latex document
    plt.plot(p0, entropy, 'b', lw=0.7)
    plt.xlabel(r'$ \mathbf{p_{0}} $')
    plt.ylabel(r'$ \mathbf{H_{2}(p_{0})\hspace{0.2}[bit]} $')
    plt.grid()
    plt.tight_layout()

    file = os.path.basename(__file__).split('.')[0]
    plt.savefig(f'../Results/{file}/entropy_vs_p0.png', dpi=800)
