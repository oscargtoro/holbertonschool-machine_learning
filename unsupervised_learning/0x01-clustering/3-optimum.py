#!/usr/bin/env python3
"""Module for the function optimum_k
"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """tests for the optimum number of clusters by variance:

    Args.
        X: is a numpy.ndarray of shape (n, d) containing the data set
        kmin: is a positive integer containing the minimum number of clusters
        to check for (inclusive)
        kmax: is a positive integer containing the maximum number of clusters
        to check for (inclusive)
        iterations: is a positive integer containing the maximum number of
        iterations for K-means

    Returns.
        A list containing the outputs of K-means for each cluster size and
        a list containing the difference in variance from the smallest cluster
        size for each cluster size, or None, None on failure
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(kmax, int) or kmax <= 0:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    outputs = []
    differences = []
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        outputs.append((C, clss))
        if k == kmin:
            vTop = variance(X, C)
        vCurrent = variance(X, C)
        differences.append(vTop - vCurrent)
    return outputs, differences
