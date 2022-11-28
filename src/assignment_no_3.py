#!/usr/bin/env python3.9.5
import numpy as np
import os
import seaborn as sns
from lib.information_lib import Entropy, Mutual_information
from lib.pmf_estimation import joint_p, pmf_univariate
from itertools import combinations
import pandas as pd


if __name__ == "__main__":
    # Load the iris dataset
    df = sns.load_dataset("iris")
    print('\n', df.info())

    print('\n', df.head(10))

    # Get a copy of the dataframe 
    df_cp = df.copy()

    # Discretizing the Dataset by multiplying by 10 all the features
    # and casting them to integers values
    df_cp.iloc[:, :-1] = (df_cp.iloc[:, :-1] * 10).astype(int)

    # Get all the feature names and its number
    features = df_cp.columns[:-1]
    n = len(features)

    # Compute the pmf and the entropy of each features   
    entropies, pmfs, x_vals = [], [], []
    for feature in features:
        # Computing the pmf
        x_val, pmf = pmf_univariate(df_cp[feature])  
        # Computing the entropy
        entropies.append(Entropy(pmf)), pmfs.append(pmf), x_vals.append(x_val)
        

    # Saving x_vals list as .csv file
    file = os.path.basename(__file__).split('.')[0]
    df = pd.DataFrame(x_vals, index=features).T
    # Replacing nan values with zeros for a matter of plot settings
    df = df.fillna(0).astype('int')

    df.to_csv(f'../Results/{file}/x_vals_1.csv', index=False)

    # Saving pmfs list as .csv file
    df = pd.DataFrame(pmfs, index=features).T
    df.to_csv(f'../Results/{file}/pmfs_2.csv', index=False)

    # Print pmfs
    print('\n', df, '\n')

    # Print entropies
    print('\n', pd.DataFrame(dict(zip(features, entropies)),
                            index=['entropy']), '\n')

    # Computing the Mutual information over all the features combinations
    mi_mat = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        feat1, feat2 = df_cp.iloc[:, i], df_cp.iloc[:, j] # extract the features
        pxy = joint_p(feat1, feat2) # Computing the Joint Probability
        px = pxy.sum(axis=1)  # Computing the x marginal probability
        py = pxy.sum(axis=0)  # Computing the y marginal probability 
        # Mutual Information is symmetric so it is enough compute only
        # the upper triangular matrix
        mi_mat[i, j] = Mutual_information(pxy, px, py)

    # Make symmetric the Mutual Information matrix 
    i_lower = np.tril_indices(n, -1)
    mi_mat[i_lower] = mi_mat.T[i_lower]

    # Fill the diagonal with the entropy values, because of the relation
    # I(X;X) = H(X)
    np.fill_diagonal(mi_mat, entropies)

    # Print mutual information matrix
    print(pd.DataFrame(mi_mat, columns=features, index=features))

    # Save the Mutual Information matrix as npy file
    np.save(f'../Results/{file}/mutual_information_matrix', mi_mat)
