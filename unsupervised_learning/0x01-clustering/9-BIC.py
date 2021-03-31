#!/usr/bin/env python3
"""Module for the function BIC
"""

expectation_maximization = __import__('7-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Finds the best number of clusters for a GMM using the Bayesian
    Information Criterion

    Args.
        X: is a numpy.ndarray of shape (n, d) containing the data set
        kmin: is a positive integer containing the minimum number of clusters
        to check for (inclusive)
        kmax: is a positive integer containing the maximum number of clusters
        to check for (inclusive)
            -If kmax is None, kmax should be set to the maximum number of
            clusters possible
        iterations: is a positive integer containing the maximum number of
        iterations for the EM algorithm
        tol: is a non-negative float containing the tolerance for the EM
        algorithm
        verbose: is a boolean that determines if the EM algorithm should print
        information to the standard output

    Returns.
        The best value for k based on its BIC, a tuple containing a
        numpy.ndarray of shape (k,) containing the cluster priors for the best
        number of clusters, a numpy.ndarray of shape (k, d) (containing the
        centroid means for the best number of clusters, a numpy.ndarray of
        shape (k, d, d) containing the covariance matrices for the best number
        of clusters a numpy.ndarray of shape (kmax - kmin + 1) containing the
        log likelihood for each cluster size tested a numpy.ndarray of shape
        (kmax - kmin + 1) containing the BIC value for each cluster size
        tested), the number of parameters required for the model, the number
        of data points used to create the model and the log likelihood of the
        model or None, None, None, None on failure
    """
    return None, None, None, None
