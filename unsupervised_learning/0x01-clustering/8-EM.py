#!/usr/bin/env python3
"""Module for the function expectation_maximization
"""

import numpy as np
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
    return None, None, None, None, None
