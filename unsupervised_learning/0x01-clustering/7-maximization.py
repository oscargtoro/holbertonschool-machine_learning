#!/usr/bin/env python3
"""Module for the function maximization
"""

import numpy as np


def maximization(X, g):
    """Calculates the maximization step in the EM algorithm for a GMM

    Args.
        X is a numpy.ndarray of shape (n, d) containing the data set
        g is a numpy.ndarray of shape (k, n) containing the posterior
        probabilities for each data point in each cluster

    Returns.
        A numpy.ndarray of shape (k,) containing the updated priors for each
        cluster, a numpy.ndarray of shape (k, d) containing the updated
        centroid means for each cluster and a numpy.ndarray of shape (k, d, d)
        containing the updated covariance matrices for each cluster,
        or None, None, None on failure
    """
    return None, None, None
