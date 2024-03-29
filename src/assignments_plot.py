import glob
import numpy as np
import pandas as pd
from scipy.stats import norm
from lib.analysis_plot import plot_settings, plot_kernels, \
                              power_10_axis_formatter
from lib.information_lib import Diff_entropy
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':

    ##############################Assignment_no_1##############################
    
    file = 'assignment_no_1'
    file_path = f'../Results/{file}/'   
    file_name = glob.glob1(file_path, '*.npy')[0]

    p0, entropy = np.load(file_path + file_name)

    plot_settings() # for latex document
    plt.plot(p0, entropy, 'b', lw=0.7)
    plt.xlabel(r'$ p_{0} $')
    plt.ylabel(r'$ H(p_{0})\hspace{0.2}[nat] $')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'../Results/{file}/entropy_vs_p0.png', dpi=800)

    print(f'{file}_plot is done. \n')

    ##############################Assignment_no_2##############################

    file = 'assignment_no_2'
    file_path = f'../Results/{file}/'
    file_name = glob.glob1(file_path, '*1.csv')[0]

    df = pd.read_csv(file_path + file_name)

    M = df.n_samples.value_counts().iloc[0]
    p0 = np.linspace(0, 1, M)

    fig, axs = plt.subplots(1, 2, constrained_layout=True)
    samples = df.n_samples.unique()

    for N in samples:
        esp = int(np.log10(N))
        axs[0].plot(p0, df.loc[df.n_samples == N, 'est_entropy'], lw=0.7,
                    label=f'$ N = 10^{esp} $')
        axs[1].plot(p0, df.loc[df.n_samples == N, 'difference'], lw=0.7)

    for i in range(2):
        axs[i].set_xlabel(r'$ p_{0} $')
        axs[i].grid()

    leg = axs[0].legend()
    for lh in leg.legendHandles: 
        lh.set_linewidth(2)

    axs[0].set_ylabel(r'$ H(p_{0})\hspace{0.2}[nat] $')
    axs[1].set_ylabel(r'$ (H(p_{0}) - H(\hat{p_{0}}))\hspace{0.2}[nat]} $')
    power_10_axis_formatter(axs[1], 'y')

    plt.savefig(f'../Results/{file}/estimated_entropy_vs_p0.png', dpi=800)
    plt.clf()
    plt.close()


    file_name = glob.glob1(file_path, '*2.csv')[0]

    df = pd.read_csv(file_path + file_name)

    kernels = df['kernel'].unique()
    methods = df['bw_method'].unique()
    n_samples = df.n_samples.unique()

    x_truth = np.linspace(-5, 5, 10000)
    pdf_truth = norm.pdf(x_truth, loc=0.0, scale=1.0)
    diff_entropy = Diff_entropy(pdf_truth, x_truth)
    
    
    fig, ax = plt.subplots(constrained_layout=True)
    plot_settings()
    plot_kernels(ax, kernels)
    plt.savefig(f'../Results/{file}/kernels.png', dpi=800)
    
    fig, axs = plt.subplots(2, 3, figsize=(10, 8), constrained_layout=True)
    axs = axs.flatten()
    plot_settings()

    kernels = df['kernel'].unique()
    methods = df['bw_method'].unique()
    n_samples = df.n_samples.unique()

    colors = ['orange', 'deepskyblue', 'green', 'red', 'grey', 'purple']

    for i, method in enumerate(methods):       
        for j, kernel in enumerate(kernels):
            tmp = df.copy()
            tmp = tmp.set_index('n_samples')
            series = tmp.loc[(tmp.bw_method == method) & \
                            (tmp['kernel'] == kernel), 'results']
            axs[i].plot(series, c=colors[j], lw=0.8, 
                        label=kernel.capitalize())

        axs[i].plot(n_samples, [diff_entropy] * len(n_samples), 'k',
                    lw=0.8, label='True pdf')
        axs[i].grid()
        axs[i].set_xscale('log')
        axs[i].set_xlabel(tmp.index.name.replace('n_', ' '))
        axs[i].set_title(method.capitalize())    
            
    legend = axs[-1].legend()
    for lh in legend.legendHandles: 
        lh.set_linewidth(2)

    plt.savefig(f'../Results/{file}/estimated_diff_entropy_vs_p0.png', dpi=800)
    
    print(f'{file}_plot is done. \n')

    ##############################Assignment_no_3##############################

    file = 'assignment_no_3'
    file_path = f'../Results/{file}/'
    file_name1 = glob.glob1(file_path, '*1.csv')[0]
    file_name2 = glob.glob1(file_path, '*2.csv')[0]

    df1 = pd.read_csv(file_path + file_name1)
    df2 = pd.read_csv(file_path + file_name2)
    features = df1.columns   

    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    plot_settings()
    axs = axs.flatten()
    labels = [feature.capitalize().replace('_', ' ') for feature in features]

    for i, feature in enumerate(features):
        mask = df1.ne(0)[feature]
        sns.barplot(x=df1.loc[mask, feature].to_numpy(),
                    y=df2.loc[mask, feature].to_numpy(),
                    ax=axs[i], color='b')
        axs[i].set_title(labels[i])
        arr = axs[i].get_xticks()
        axs[i].set_xticks(arr[::4])

    plt.savefig(f'../Results/{file}/estimated_pmf_features.png', dpi=800)

    print(f'{file}_plot is done. \n')
