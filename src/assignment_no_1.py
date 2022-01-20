#!/usr/bin/env python3.9.5
import sys
import os
from lib.utils import *
import numpy as np
import matplotlib.pyplot as plt
from lib.analysis_plot import *
from lib.information_lib import entropy


if __name__ == '__main__':
    # Read the input from the terminal for the size of random data (N) 
    # and the number of samples to generate for p0 (M)
    if len(sys.argv) != 3:
        print("N (10000), M (1000)")
    else:
        M, N = np.int32(sys.argv[1:])
        
        p0 = np.linspace(0, 1, M)
        x = [random_data_gen(p, N) for p in p0]
        pk = list(map(prob, x))
        h = list(map(entropy, pk))
        
        plot_settings() # for latex document
        plt.plot(p0, h, 'b-', lw=0.5)
        plt.xlabel(r'$ \mathbf{p_{0}} $')
        plt.ylabel(r'$ \mathbf{H_{2}(p_{0})\hspace{0.2}[bit]} $')
        #plt.title("Entropy for a generic binary random variable",
                  #weight='bold')
        plt.tight_layout()
        file = os.path.basename(__file__).split('.')[0]
        plt.savefig(f'../Results/{file}/entropy_vs_p0.png', dpi=800)
        #plt.show()
