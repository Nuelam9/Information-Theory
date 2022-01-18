import numpy as np
import seaborn as sns
from sklearn import metrics
from lib.classifier_models import *
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    df = sns.load_dataset("iris")

    # Work on a copy of the dataframe
    df_cp = df.copy()

    # Encode the species from (setosa, versicolor, virginica) to (0, 1, 2)
    codes = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
    # Save the encoding to go back
    decode = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

    # Apply the econding for the for the species class
    df_cp['species'] = df_cp['species'].map(codes)

    X, y = df_cp.to_numpy()[:, :-1], df_cp.to_numpy()[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        train_size=0.5,  # 50% of train and test set
                                                        random_state=12, # randomly shuffling instances
                                                        stratify=y) # stratify with respect to the class distribution in y

    # Check if the train and test sets are evenly distributed
    print(np.unique(y_train, return_counts=True),
          np.unique(y_test, return_counts=True))

    # Instantiating a Bayes classifier
    BC = Bayes_Classifier()
    # Fit the data with the model
    BC.fit(X_train, y_train)
    # Get the prediction on the test set
    BC_pred = BC.evaluate(X_test)
    # Compute the model acccuracy
    BC_acc = metrics.accuracy_score(BC_pred, y_test)
