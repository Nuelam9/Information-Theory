import numpy as np 
from statsmodels.nonparametric.kernel_density import KDEMultivariate, KDEMultivariateConditional
from lib.utils import Gaussian


def Bayes(X_train, X_test, y_train):
    n_X = X_train.shape[1]
    n_classes = np.unique(y_train).size

    # Computing the prior distribution
    Prior = np.unique(y_train, return_counts=True)[1] / y_train.size

    # Computing P(X|C)
    P_Xc_train = KDEMultivariateConditional(endog=X_train,
                                            exog=y_train,
                                            dep_type='cccc',
                                            indep_type='u',
                                            bw=None).pdf

    # Computing the joint probability P(X)
    P_X_train = KDEMultivariate(data=X_train, var_type='cccc').pdf

    n = X_test.shape[0]
    # Computing the conditional probability P(X|c) for the new data 
    PXc = np.array([P_Xc_train(X_test, [i] * n) for i in range(n_classes)])
    # Computing the conditional probability P(c|X) = P(X|c) * P(c) / P(X)
    PcX = ((PXc.T * Prior).T / P_X_train(X_test)).T

    # Returning the class abel prediction
    return np.argmax(PcX, axis=1)


def Naive_Bayes(X_train, X_test, y_train):
        n_X = X_train.shape[1]
        n_classes = np.unique(y_train).size

        # Computing the prior distribution
        Prior = np.unique(y_train, return_counts=True)[1] / y_train.size

        P_Xc_train = []
        # Computing the P(X_i|c) for each feature
        for i in range(n_X):
            P_Xc_train.append(KDEMultivariateConditional(endog=X_train[:, i],
                                                         exog=y_train, 
                                                         dep_type='c', 
                                                         indep_type='u', 
                                                         bw='normal_reference').pdf)
        # Computing the joint probability P(X)
        P_X_train = KDEMultivariate(X_train, var_type='cccc').pdf
        
        n = X_test.shape[0]
        c = np.zeros((n, n_classes))
        
        for d in range(n):
            for i in range(n_classes):
                tmp = 1
                # Computing the P(X|c) = product of P(x_i|c)
                for j in range(n_X):
                    tmp *= P_Xc_train[j]([X_test[d, j]], [i])
                
                # Computing the conditional probability P(c|X) = P(X|c) * P(c) / P(X)
                c[d, i] = tmp * Prior[i] / P_X_train(X_test[d, :])
        
        # Returning the class prediction
        return np.argmax(c, axis=1)
    

def Gaussian_Naive_Bayes(X_train, X_test, y_train):
    n_X = X_train.shape[1]
    n_classes = np.unique(y_train).size
    classes = np.unique(y_train)
    
    # Compute the prior distribution
    Prior = np.unique(y_train, return_counts=True)[1] / y_train.size
    
    mean = np.zeros((n_classes, n_X))
    var = np.zeros((n_classes, n_X))

    # Compute mean and variance of all features
    for i, c in enumerate(classes):
        mean[i] = np.mean(X_train[y_train == c], axis=0)
        var[i] = np.var(X_train[y_train == c], axis=0)
    
    n_rows = X_test.shape[0]
    # Compute the posterior
    posteriors = np.zeros((n_rows, n_classes))
    
    for i in range(n_classes):
        prior = np.log(Prior[i])
        # Computing the log probability of P(X|c)
        conditional = np.sum(np.log(Gaussian(X_test, mean[i], var[i])), axis=1)
        # Computing the P(c|X) = P(X|c) * P(c) / P(X)
        posteriors[:, i] = prior + conditional
    
    # returning the class predictions
    return np.argmax(posteriors, axis=1)
