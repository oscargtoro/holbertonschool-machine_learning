#!/usr/bin/env python3
"""Module for the function expectation_maximization
"""

import numpy as np
from numpy.lib.arraysetops import isin
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """Performs the expectation maximization for a GMM

    Args.
        X: is a numpy.ndarray of shape (n, d) containing the data set
        k: is a positive integer containing the number of clusters
        iterations: is a positive integer containing the maximum number of
        iterations for the algorithm
        tol: is a non-negative float containing tolerance of the log
        likelihood, used to determine early stopping i.e. if the difference is
        less than or equal to tol you should stop the algorithm
        verbose: is a boolean that determines if you should print information
        about the algorithm

    Returns:
        A numpy.ndarray of shape (k,) containing the priors for each cluster,
        a numpy.ndarray of shape (k, d) containing the centroid means for each
        cluster, a numpy.ndarray of shape (k, d, d) containing the covariance
        matrices for each cluster, a numpy.ndarray of shape (k, n) containing
        the probabilities for each data point in each cluster and the log
        likelihood of the model, or None, None, None, None, None on failure
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    g, logLikelihood = expectation(X, pi, m, S)
    prevLikelihood = 0
    message = "Log Likelihood after {} iterations: {}"

    for i in range(iterations):
        if verbose and i % 10 == 0:
            print(message.format(i, logLikelihood.round(5)))

        pi, m, S = maximization(X, g)
        g, logLikelihood = expectation(X, pi, m, S)

        if abs(prevLikelihood - logLikelihood) <= tol:
            break

        prevLikelihood = logLikelihood

    if verbose:
        print(message.format(i + 1, logLikelihood.round(5)))

    return pi, m, S, g, logLikelihood
