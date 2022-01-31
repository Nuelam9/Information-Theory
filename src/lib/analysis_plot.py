#!/usr/bin/env python3.9.5
import numpy as np


def plot_settings():
    import matplotlib as mpl
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


def power_10_axis_formatter(ax, axis):
    from matplotlib import ticker
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True) 
    formatter.set_powerlimits((-1, 1))
    if axis == 'x':
        ax.xaxis.set_major_formatter(formatter) 
    elif axis == 'y':
        ax.yaxis.set_major_formatter(formatter) 


def plot_kernels(ax, kernels):
    """Visualize the KDE kernels"""
    from sklearn.neighbors import KernelDensity
    ax.grid()

    X_src = np.zeros((1, 1))
    x_grid = np.linspace(-3, 3, 1000)

    for kernel in kernels:
        kde = KernelDensity(kernel=kernel).fit(X_src)
        log_dens = kde.score_samples(x_grid[:, None])
        ax.plot(x_grid, np.exp(log_dens), lw=0.7, label=kernel.capitalize())
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-2.9, 2.9)
    ax.set_xlabel(r'$ x $')
    ax.set_title('Kernels')
    leg = ax.legend()
    for lh in leg.legendHandles: 
        lh.set_linewidth(2)
    