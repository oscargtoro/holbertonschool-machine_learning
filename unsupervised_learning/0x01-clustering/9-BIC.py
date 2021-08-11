#!/usr/bin/env python3
"""Module for the function BIC
"""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Finds the best number of clusters for a GMM using the Bayesian
    Information Criterion

    Args.
        X: is a numpy.ndarray of shape (n, d) containing the data set
        kmin: is a positive integer containing the minimum number of clusters
        to check for (inclusive)
        kmax: is a positive integer containing the maximum number of clusters
        to check for (inclusive)
        iterations: is a positive integer containing the maximum number of
        iterations for the EM algorithm
        tol: is a non-negative float containing the tolerance for the EM
        algorithm
        verbose: is a boolean that determines if the EM algorithm should print
        information to the standard output

    Returns.
        The best value for k based on its BIC, a tuple containing:
            - a numpy.ndarray of shape (k,) containing the cluster priors for
            the best number of clusters
            - a numpy.ndarray of shape (k, d) containing the centroid means for
            the best number of clusters
            - a numpy.ndarray of shape (k, d, d) containing the covariance
            matrices for the best number of clusters
        A numpy.ndarray of shape (kmax - kmin + 1) containing the log
        likelihood for each cluster size tested, a numpy.ndarray of shape
        (kmax - kmin + 1) containing the BIC value for each cluster size tested
        or None, None, None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None
    if isinstance(kmax, type(None)):
        kmax = X.shape[0]
    elif not isinstance(kmax, int) or kmax <= 0:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape
    best_k = []
    best_result = []
    logLikelihood = []
    b = []

    for k in range(kmin, kmax + 1):
        best_k.append(k)
        pi, m, S, _, kloglikelihood = expectation_maximization(X,
                                                               k,
                                                               iterations,
                                                               tol,
                                                               verbose)
        best_result.append((pi, m, S))
        logLikelihood.append(kloglikelihood)
        p = (k * d * (d + 1) / 2) + (d * k) + k - 1
        BIC = p * np.log(n) - 2 * kloglikelihood
        b.append(BIC)

    logLikelihood = np.array(logLikelihood)
    index = np.argmin(b)

    return best_k[index], best_result[index], logLikelihood, np.array(b)
