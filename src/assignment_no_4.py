#!/usr/bin/env python3.9.5
import numpy as np
import seaborn as sns
from sklearn import metrics
from lib.classifiers import *
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    df = sns.load_dataset("iris")
    
    # Work on a copy of the dataframe
    df_cp = df.copy()

    # Encode the species from (setosa, versicolor, virginica) to (0, 1, 2)
    codes = {'setosa': 0, 'versicolor': 1, 'virginica': 2}

    # Apply the econding for the for the species class
    df_cp['species'] = df_cp['species'].map(codes)

    # Convert features columns in numpy arrays   
    X = df_cp.to_numpy()[:, :-1]
    # Convert the class label column in an integer numpy array
    y = df_cp.to_numpy(dtype=int)[:, -1]

    # Data is divided in a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        train_size=0.5,  # 50% of train and test set
                                                        random_state=12, # randomly shuffling instances
                                                        stratify=y) # stratify with respect to the class distribution in y

    # Check if the train and test sets are evenly distributed
    print(np.unique(y_train, return_counts=True),
          np.unique(y_test, return_counts=True))

    # Apply a Bayes classifier and get the prediction on the test set with
    B_prediction = Bayes(X_train, X_test, y_train)
    # Compute the model acccuracy
    B_acc = metrics.accuracy_score(B_prediction, y_test)
    # Apply a Naive Bayes classifier and get the prediction on the test set with
    NB_prediction = Naive_Bayes(X_train, X_test, y_train)
    # Compute the model acccuracy
    NB_acc = metrics.accuracy_score(NB_prediction, y_test)

    # Apply a Naive Bayes classifier and get the prediction on the test set with
    GNB_prediction = Gaussian_Naive_Bayes(X_train, X_test, y_train)
    # Compute the model acccuracy
    GNB_acc = metrics.accuracy_score(GNB_prediction, y_test)

    # Print all the models accuracy
    print(f'Bayes, Naive Bayes, Naive Bayes Gaussian: {B_acc}, \
                                                      {NB_acc}, \
                                                      {GNB_acc}')
