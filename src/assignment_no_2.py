#!/usr/bin/env python3.9.5
import numpy as np
import os
from lib.information_lib import Entropy, Diff_entropy
from lib.pdf_estimation import kde_estimate
import matplotlib.pyplot as plt
from lib.utils import random_data_gen, prob
from lib.analysis_plot import *
from scipy.stats import norm
import pandas as pd


if __name__ == '__main__': 
    # Set the number of samples to generate for p0 (M)
    M = 100
    p0 = np.linspace(0, 1, M)
    p = np.vstack([p0, 1 - p0]).T

    # Computing the Entropy of the truth p.m.f.
    entropy = list(map(Entropy, p))
         
    plot_settings() # for latex document
    fig, axs = plt.subplots(1, 2, constrained_layout=True)
    axs[0].plot(p0, entropy, 'b', lw=0.7, label='truth p.m.f.')
    
    # Cycle over the size of random data (N)
    for N in 10 ** np.arange(2, 6):
        # Computing the entropy for the p0 values contained in p array
        x = [random_data_gen(p, N) for p in p0]
        # Compute the estimated pmf
        pk = list(map(prob, x))
        # Compute the entropy from the estimated p.m.f.
        est_entropy = list(map(Entropy, pk))
        entropy_diff = [h - h_est for h, h_est in zip(entropy, est_entropy)]
        esp = int(np.log10(N))
        axs[0].plot(p0, est_entropy, lw=0.7, 
                    label=f'N=$ 10^{esp} $')
        axs[1].plot(p0, entropy_diff, lw=0.7)
    
    for i in range(2):
        axs[i].set_xlabel(r'$ \mathbf{p_{0}} $')
        axs[i].grid()
    
    leg = axs[0].legend()
    for lh in leg.legendHandles: 
        lh.set_linewidth(2)

    axs[0].set_ylabel(r'$ \mathbf{H_{2}(p_{0})\hspace{0.2}[bit]} $')
    axs[1].set_ylabel(r'$ \mathbf{(H_{2}(p_{0})-H_{2}(\hat{p_{0}}))\hspace{0.2}[bit]} $')
    power_10_axis_formatter(axs[1], 'y')

    file = os.path.basename(__file__).split('.')[0]
    plt.savefig(f'../Results/{file}/estimated_entropy_vs_p0.png', dpi=800)
    plt.clf()
    plt.close()

   ########################Differential entropy#########################
    
    x_truth = np.linspace(-5, 5, 10000)
    pdf_truth = norm.pdf(x_truth)
    diff_entropy = Diff_entropy(pdf_truth, x_truth)

    n_generated = 10000  # samples number to generate
    E_pdf = []
    kernels = ['gaussian', 'tophat', 'epanechnikov',
               'exponential', 'linear', 'cosine']

    n_samples = 10 ** np.arange(2, 7)
    results = np.zeros((len(n_samples), len(kernels)))
    
    for i, kernel in enumerate(kernels):
        diff_E = []
        for j, n in enumerate(n_samples):
            samples = np.random.normal(size=n)
            x, est_pdf = kde_estimate(samples, kernel)
            diff_E.append(Diff_entropy(est_pdf, x))

        results[:, i] = diff_E

    df = pd.DataFrame({'n samples': n_samples})

    df[kernels] = results
    df = df.set_index('n samples')
    print(df)

    plot_settings() # for latex document
    fig, axs = plt.subplots(1,2, constrained_layout=True) 
    df.plot(ax=axs[0], logx=True, lw=0.5)
    axs[0].plot(n_samples, [diff_entropy] * len(n_samples),
                label='truth pdf', lw=0.5)
    (diff_entropy - df).plot(ax=axs[1], logx=True, lw=0.5, legend=False)
    legend = axs[0].legend()
    legend.get_frame().set_alpha(0.5)

    for i in range(2):
        axs[i].set_xlabel(r'$ \mathbf{n° samples} $')
        axs[i].grid()
    
    axs[1].set_xlabel(r'$ \mathbf{n° samples} $')
    axs[0].set_ylabel(r'$ \mathbf{h_{2}(p_{0})} $')
    axs[1].set_ylabel(r'$ \mathbf{h_{2}(p_{0}) - h_{2}(\hat{p_{0}})} $')
    plt.savefig(f'../Results/{file}/estimated_diff_entropy_vs_p0.png', dpi=800)
