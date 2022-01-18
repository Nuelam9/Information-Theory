#!/usr/bin/env python3.9.5
import numpy as np
import os
import seaborn as sns
from lib.information_lib import entropy, mutual_information
from lib.pmf_estimation import joint_p, pmf_univariate
from itertools import combinations


if __name__ == "__main__":
    # Load the iris dataset
    df = sns.load_dataset("iris")
    print(df.info())

    # Get a copy of the dataframe 
    df_cp = df.copy()

    # Discretizing the Dataset by multiplying by 10 all the features
    # and casting them to integers values
    df_cp.iloc[:, :-1] = (df_cp.iloc[:, :-1] * 10).astype(int)

    # Get all the feature names and its number
    features = df_cp.columns[:-1]
    n = len(features)

    # Compute the pmf and the entropy of each features   
    entropies = []
    for feature in features:
        x_vals, pmf = pmf_univariate(df_cp[feature])  # Computing the pmf
        entropies.append(entropy(pmf))

    # Computing the Mutual information over all the features combinations
    mi_mat = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        feat1, feat2 = df_cp.iloc[:, i], df_cp.iloc[:, j] # extract the features
        pxy = joint_p(feat1, feat2) # Computing the Joint Probability
        px = pxy.sum(axis=1)  # Computing the x marginal probability
        py = pxy.sum(axis=0)  # Computing the y marginal probability 
        # Mutual Information is symmetric so it is enough compute only
        # the upper triangular matrix
        mi_mat[i, j] = mutual_information(pxy, px, py)

    # Make symmetric the Mutual Information matrix 
    i_lower = np.tril_indices(n, -1)
    mi_mat[i_lower] = mi_mat.T[i_lower] 

    # Save the Mutual Information matrix as npy file
    file = os.path.basename(__file__).split('.')[0]
    np.save(f'../results/{file}/mutual_information_matrix', mi_mat)
