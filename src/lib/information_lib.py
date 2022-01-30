#!/usr/bin/env python3.9.5
import numpy as np
from scipy.special import xlogy
from scipy.integrate import simpson 
#import warnings

#warnings.filterwarnings("ignore")


def Entropy(px: np.array) -> float:
    """Compute the entropy of a discrete random variable given its 
       probability mass function px = [p1, p2, ..., pN]

    Arg:
        px (np.array): p.m.f.

    Return:
        float: entropy (nats units)
    """
    return - np.sum(xlogy(px, px))
    

def Joint_entropy(pxy: np.ndarray) -> float:
    """Compute the joint entropy of two generic discrete random 
       variables given their joint p.m.f.

    Arg:
        pxy (np.ndarray): joint p.m.f.

    Return:
        float: joint entropy (nats units)
    """
    return - np.sum(xlogy(pxy, pxy))


def Cond_entropy(pxy: np.ndarray, py: np.array) -> float:
    """Compute the conditional entropy of two generic discrete random 
       variables given their joint and marginal p.m.f., using the 
       relation:
                 H(X|Y) = H(X,Y) - H(Y) 

    Args:
        pxy (np.ndarray): joint p.m.f.
        py (np.array): marginal p.m.f. of r.v. Y

    Return:
        float: conditional entropy (nats units)
    """
    return Joint_entropy(pxy) - Entropy(py) 


def Mutual_information(pxy: np.ndarray, px: np.array, py: np.array) -> float:
    """Compute the mutual information of two generic discrete random 
       variables given their joint and marginal p.m.f., using the 
       relation:
                I(X;Y) = H(X) + H(Y) - H(X,Y)

    Args:
        pxy (np.ndarray): joint p.m.f.
        px (np.array): marginal p.m.f. of r.v. X
        py (np.array): marginal p.m.f. of r.v. Y

    Return:
        float: mutual information (nats units)
    """
    return Entropy(px) + Entropy(py) - Joint_entropy(pxy)


def Norm_cond_entropy(pxy: np.ndarray, px: np.array) -> float:
    """Compute the normalized conditional entropy of two generic 
       discrete random variables given their joint and marginal p.m.f.,
       using the relation:
                          eta_CE(X|Y) = H(X|Y) / H(X)
       and it is bounded in the interval [0, 1].

    Args:
        pxy (np.ndarray): joint p.m.f.
        px (np.array): marginal p.m.f. of r.v. X

    Return:
        float: normalized conditional entropy (nats units)
    """
    return Cond_entropy(pxy, px) / Entropy(px)


def Norm_joint_entropy(pxy: np.ndarray, px: np.array, py: np.array) -> float:
    """Compute the normalized joint entropy of two generic discrete 
       random variables given their joint and marginal p.m.f., using the 
       formula:
               eta_JE(X,Y) = 1 - I(X;Y) / (H(X) + H(Y)) 
       and it is bounded in the interval [1/2, 1].

    Args:
        pxy (np.ndarray): joint p.m.f.
        px (np.array): marginal p.m.f. of r.v. X
        py (np.array): marginal p.m.f. of r.v. Y

    Return:
        float: normalized joint entropy (nats units)
    """
    return 1. - Mutual_information(pxy, px, py) / (Entropy(px) + Entropy(py))


def Norm_mutual_information1(pxy: np.ndarray, px: np.array, py: np.array) -> float:
    """Compute the normalized mutual information of type 1 of two 
       generic discrete random variables given their joint and marginal 
       p.m.f., using the relation:
                    eta_MI1(X;Y) = 1 / eta_JE(X,Y) - 1
       and it is bounded in the interval [0, 1].

    Args:
        pxy (np.ndarray): joint p.m.f.
        px (np.array): marginal p.m.f. of r.v. X
        py (np.array): marginal p.m.f. of r.v. Y

    Return:
        float: normalized mutual information of type 1 (nats units)
    """
    return 1. / Norm_joint_entropy(pxy, px, py) - 1.


def Norm_mutual_information2(pxy: np.ndarray, px: np.array, py: np.array) -> float:
    """Compute the normalized mutual information of type 2 of two 
       generic discrete random variables given their joint and marginal 
       p.m.f., using the relation:
                    eta_MI2(X;Y) = 1 + eta_MI1(X;Y)
       and it is bounded in the interval [1, 2].

    Args:
        pxy (np.ndarray): joint p.m.f.
        px (np.array): marginal p.m.f. of r.v. X
        py (np.array): marginal p.m.f. of r.v. Y

    Return:
        float: normalized mutual information of type 2 (nats units)
    """
    return 1. + Norm_mutual_information1(pxy, px, py)


def Norm_mutual_information3(pxy: np.ndarray, px: np.array, py: np.array) -> float:
    """Compute the normalized mutual information of type 3 of two 
       generic discrete random variables given their joint and marginal 
       p.m.f., using the relation:
                    eta_MI3(X;Y) = I(X;Y) / sqrt(H(X) * H(Y))
       and it is bounded in the interval [0, 1].

    Args:
        pxy (np.ndarray): joint p.m.f.
        px (np.array): marginal p.m.f. of r.v. X
        py (np.array): marginal p.m.f. of r.v. Y

    Return:
        float: normalized mutual information of type 3 (nats units)
    """
    return Mutual_information(pxy, px, py) / np.sqrt(Entropy(px) * Entropy(py))


def Diff_entropy(fx: np.array, x: np.array) -> float:
    """Compute the differential entropy using the following formula:
       h(X) = -\int_{a}^{b} f_{X}(x) \ ln(f_{X}(x)) dx

    Args:
        fx (np.array): pdf of the continuos random variable
        x (np.array): the points at which fx is sampled
    """
    integrand = - xlogy(fx, fx)
    # Use the simpson integral method
    return simpson(y=integrand, x=x)
