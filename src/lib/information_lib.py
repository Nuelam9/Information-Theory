#!/usr/bin/env python3.9.5
import numpy as np
from scipy.integrate import simpson 


def entropy(px: np.array) -> float:
    """Compute the entropy of a discrete random variable given its 
       probability mass function px = [p1, p2, ..., pN]

    Arg:
        px (np.array): p.m.f.

    Return:
        float: entropy (bits units)
    """
    return - np.nansum(px * np.log2(px))


def joint_entropy(pxy: np.ndarray) -> float:
    """Compute the joint entropy of two generic discrete random 
       variables given their joint p.m.f.

    Arg:
        pxy (np.ndarray): joint p.m.f.

    Return:
        float: joint entropy (bits units)
    """
    return - np.nansum(pxy * np.log2(pxy))


def conditional_entropy(pxy: np.ndarray, py: np.array) -> float:
    """Compute the conditional entropy of two generic discrete random 
       variables given their joint and marginal p.m.f., using the 
       relation:
                 H(X|Y) = H(X,Y) - H(Y) 

    Args:
        pxy (np.ndarray): joint p.m.f.
        py (np.array): marginal p.m.f. of r.v. Y

    Return:
        float: conditional entropy (bits units)
    """
    return joint_entropy(pxy) - entropy(py) 


def mutual_information(pxy: np.ndarray, px: np.array, py: np.array) -> float:
    """Compute the mutual information of two generic discrete random 
       variables given their joint and marginal p.m.f., using the 
       relation:
                I(X,Y) = H(X) + H(Y) - H(X|Y)

    Args:
        pxy (np.ndarray): joint p.m.f.
        px (np.array): marginal p.m.f. of r.v. X
        py (np.array): marginal p.m.f. of r.v. Y

    Return:
        float: mutual information (bits units)
    """
    return entropy(px) + entropy(py) - joint_entropy(pxy)


def norm_cond_entropy(pxy: np.ndarray, px: np.array) -> float:
    """Compute the normalized conditional entropy of two generic 
       discrete random variables given their joint and marginal p.m.f.,
       using the relation:
                          eta_CE(X|Y) = H(X|Y) / H(X)
       and it is bounded in the interval [0, 1].

    Args:
        pxy (np.ndarray): joint p.m.f.
        px (np.array): marginal p.m.f. of r.v. X

    Return:
        float: normalized conditional entropy (bits units)
    """
    return conditional_entropy(pxy, px) / entropy(px)


def norm_joint_entropy(pxy: np.ndarray, px: np.array, py: np.array) -> float:
    """Compute the normalized joint entropy of two generic discrete 
       random variables given their joint and marginal p.m.f., using the 
       formula:
               eta_JE(X,Y) = 1 - I(X,Y) / (H(X) + H(Y)) 
       and it is bounded in the interval [1/2, 1].

    Args:
        pxy (np.ndarray): joint p.m.f.
        px (np.array): marginal p.m.f. of r.v. X
        py (np.array): marginal p.m.f. of r.v. Y

    Return:
        float: normalized joint entropy (bits units)
    """
    return 1. - mutual_information(pxy, px, py) / (entropy(px) + entropy(py))


def norm_mutual_information1(pxy: np.ndarray, px: np.array, py: np.array) -> float:
    """Compute the normalized mutual information of type 1 of two 
       generic discrete random variables given their joint and marginal 
       p.m.f., using the relation:
                    eta_MI1(X,Y) = 1 / eta_JE(X,Y) - 1
       and it is bounded in the interval [0, 1].

    Args:
        pxy (np.ndarray): joint p.m.f.
        px (np.array): marginal p.m.f. of r.v. X
        py (np.array): marginal p.m.f. of r.v. Y

    Return:
        float: normalized mutual information of type 1 (bits units)
    """
    return 1. / norm_joint_entropy(pxy, px, py) - 1.


def norm_mutual_information2(pxy: np.ndarray, px: np.array, py: np.array) -> float:
    """Compute the normalized mutual information of type 2 of two 
       generic discrete random variables given their joint and marginal 
       p.m.f., using the relation:
                    eta_MI2(X,Y) = 1 + eta_MI1(X,Y)
       and it is bounded in the interval [1, 2].

    Args:
        pxy (np.ndarray): joint p.m.f.
        px (np.array): marginal p.m.f. of r.v. X
        py (np.array): marginal p.m.f. of r.v. Y

    Return:
        float: normalized mutual information of type 2 (bits units)
    """
    return 1. + norm_mutual_information1(pxy, px, py)


def norm_mutual_information3(pxy: np.ndarray, px: np.array, py: np.array) -> float:
    """Compute the normalized mutual information of type 3 of two 
       generic discrete random variables given their joint and marginal 
       p.m.f., using the relation:
                    eta_MI3(X,Y) = I(X,Y) / sqrt(H(X) * H(Y))
       and it is bounded in the interval [0, 1].

    Args:
        pxy (np.ndarray): joint p.m.f.
        px (np.array): marginal p.m.f. of r.v. X
        py (np.array): marginal p.m.f. of r.v. Y

    Return:
        float: normalized mutual information of type 3 (bits units)
    """
    return mutual_information(pxy, px, py) / np.sqrt(entropy(px) * entropy(py))


def diff_entropy(fx: np.array, dx: np.array) -> float:
    """Compute the differential entropy using the following formula:
       h(X) = -\int_{a}^{b}f_{X}(x)\ln(f_{X}(x))dx

    Args:
        fx (np.array): pdf of the continuos random variable
        dx (np.array): the points at which fx is sampled
    """
    integrand = - fx * np.log(fx)
    # use the simpson integral method
    return simpson(y=integrand, x=dx)
