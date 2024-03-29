#!/usr/bin/env python3
"""Module for the function initialize
"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Initializes variables for a Gaussian Mixture Model

    Args.
        X: is a numpy.ndarray of shape (n, d) containing the data set
        k: is a positive integer containing the number of clusters

    Returns:
        A numpy.ndarray of shape (k,) containing the priors for each cluster
        (initialized evenly), a numpy.ndarray of shape (k, d) containing the
        centroid means for each cluster (initialized with K-means) and a
        numpy.ndarray of shape (k, d, d) containing the covariance matrices
        for each cluster (initialized as identity matrices),
        or None, None, None on failure
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None

    d = X.shape[1]
    m, _ = kmeans(X, k)
    pi = np.full((k, ), 1 / k)
    S = np.full((k, d, d), np.eye(d))

    return pi, m, S
