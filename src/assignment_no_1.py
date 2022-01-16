#!/usr/bin/env python3.9.5
import sys
import os
from lib.utils import *
import numpy as np
import matplotlib.pyplot as plt
from lib.information_lib import entropy


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("N (10000), M (1000), inf (0), sup (1)")
    else:
        M, N, inf, sup = np.int32(sys.argv[1:])
    
        p0 = np.linspace(inf, sup, M)
        x = [random_data_gen(_, inf, sup, N) for _ in p0]
        pk = [prob(_, inf, sup) for _ in x]
        h = list(map(entropy, pk))
        
        #plt.figure(figsize=(15, 10))
        plt.plot(p0, h, 'b-')
        plt.xlabel(r'$ \mathbf{p_{0}} $', fontsize=14)
        plt.ylabel(r'$ \mathbf{H_{2}(p_{0})\hspace{0.2}[bit]} $', fontsize=14)
        plt.title("Entropy for a generic binary random variable",
                  weight='bold', fontsize=16)
        file = os.path.basename(__file__).split('.')[0]
        plt.savefig(f'../Results/{file}/entropy_vs_p0.png', dpi=800)
        #plt.show()
