#!/usr/bin/env python3
"""Module for the function variance
"""

import numpy as np


def variance(X, C):
    """calculates the total intra-cluster variance for a data set:

    Args.
        X is a numpy.ndarray of shape (n, d) containing the data set
        C is a numpy.ndarray of shape (k, d) containing the centroid means for
        each cluster
    Returns:
        The total variance or None on failure

    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(X.shape) != 2:
        return None
    if C.shape[1] != X.shape[1]:
        return None
    distance = ((X - C[:, np.newaxis])**2).sum(axis=2)
    return np.sum(np.min(distance, axis=0))
