import sys
import numpy as np
import matplotlib.pyplot as plt
from information_lib import entropy


def random_data_gen(x: float) -> np.array:
    """
    generate random numbers between 
    inf and sup with a corresponding probability
    """
    return np.random.choice(a=[inf, sup], size=N, p=[x, 1 - x])


def prob(x: float) -> np.array:
    return np.array([len(x[x == inf]) / len(x),
                     len(x[x == sup]) / len(x)])


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("N (10000), M (1000), inf (0), sup (1)")
    else:
        N = int(sys.argv[1])
        M = int(sys.argv[2])
        inf = int(sys.argv[3])
        sup = int(sys.argv[4])    
    
        p0 = np.linspace(inf, sup, M)
        x =  list(map(random_data_gen, p0))
        pk = list(map(prob, x))
        h = list(map(entropy, pk))
        
        #plt.figure(figsize=(15, 10))
        plt.plot(p0, h, 'b-')
        plt.xlabel(r'$ \mathbf{p_{0}} $', fontsize=14)
        plt.ylabel(r'$ \mathbf{H_{2}(p_{0})\hspace{0.2}[bit]} $', fontsize=14)
        plt.title("Entropy for a generic binary random variable",
                  weight='bold', fontsize=16)
        plt.savefig('./Results/entropy_vs_p0.png', dpi=400)
        #plt.show()
