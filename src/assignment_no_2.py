#!/usr/bin/env python3.9.5
import numpy as np
import os
from lib.information_lib import diff_entropy
from lib.pdf_estimation import kde_estimate
import matplotlib.pyplot as plt
from lib.analysis_plot import *
from scipy.stats import norm
import pandas as pd


if __name__ == '__main__':
    xReal = np.linspace(-5, 5, 10000)
    pdfReal = norm.pdf(xReal)

    RealEntropy = diff_entropy(pdfReal, xReal)

    n_generated = 10000  # samples number to generate
    E_pdf = []
    kernels = ['gaussian', 'tophat', 'epanechnikov',
               'exponential', 'linear', 'cosine']

    n_samples = [np.power(10, i) for i in range(1, 6)]
    results = np.zeros((len(n_samples), len(kernels)))
    for i, kernel in enumerate(kernels):
        E_pdf = []
        for j, n in enumerate(n_samples):
            samples = np.random.normal(size=n)
            x, pdfEstimated = kde_estimate(samples, kernel)
            E_pdf.append(diff_entropy(pdfEstimated, x))

        print(kernel)
        results[:, i] = E_pdf

    df = pd.DataFrame({'N generated': n_samples})

    df[kernels] = results
    df = df.set_index('N generated')
    print(df)

    #plot_settings() # for latex document
    fig, axs = plt.subplots(1, 2)
    plot_kernels(axs[0])

    df.plot(logx=True, ax=axs[1], lw=0.5)
    axs[1].plot(n_samples, [diff_entropy(pdfReal, xReal)] * len(n_samples),
                label='exact pdf', lw=0.5)
    axs[1].legend()
    axs[1].grid()
    axs[1].set_title('Differential Entropy')
    fig.tight_layout()
    plt.show()
    file = os.path.basename(__file__).split('.')[0]
    #plt.savefig(f'../results/{file}/Differential_entropy.png', dpi=600)

    print(f'The real entropy is: {RealEntropy}')