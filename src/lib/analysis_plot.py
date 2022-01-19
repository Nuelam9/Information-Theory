import matplotlib as mpl
import numpy as np
from sklearn.neighbors import KernelDensity


def plot_settings():
    fig_width_pt = 390.0    # Get this from LaTeX using \the\columnwidth
    inches_per_pt = 1.0 / 72.27                # Convert pt to inches
    golden_mean = (np.sqrt(5) - 1.0) / 2.0     # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt   # width in inches
    fig_height = fig_width * golden_mean       # height in inches
    fig_size = [fig_width, fig_height]
    params = {'backend': 'ps',
              'axes.labelsize': 14,
              'legend.fontsize': 9,
              'xtick.labelsize': 10,
              'ytick.labelsize': 10,
              'figure.figsize': fig_size,  
              'axes.axisbelow': True}

    mpl.rcParams.update(params)


def plot_kernels(ax):
    """Visualize the KDE kernels available in Scikit-learn"""
    ax.grid()

    X_src = np.zeros((1, 1))
    x_grid = np.linspace(-3, 3, 1000)

    for kernel in ['gaussian', 'tophat', 'epanechnikov',
                   'exponential', 'linear', 'cosine']:
        log_dens = KernelDensity(kernel=kernel).fit(X_src).score_samples(x_grid[:, None])
        ax.plot(x_grid, np.exp(log_dens), lw=0.5, label=kernel)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-2.9, 2.9)
    ax.set_xlabel('x')
    #ax.legend()
    ax.set_title('Kernels')