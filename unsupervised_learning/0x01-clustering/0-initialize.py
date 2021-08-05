#!/usr/bin/env python3
"""Module for the function initialize
"""

import numpy as np


def initialize(X, k):
    """Initializes cluster centroids for K-means

    Args.
        X: is a numpy.ndarray of shape (n, d) containing the dataset that will
        be used for K-means clustering
            -n: is the number of data points
            -d: is the number of dimensions for each data point
        k: is a positive integer containing the number of clusters
    Returns:
        A numpy.ndarray of shape (k, d) containing the initialized centroids
        for each cluster, or None on failure
    """
    if len(X.shape) != 2:
        return None
    if k > X.shape[0]:
        return None
    return np.random.uniform(X.min(axis=0), X.max(axis=0), (k, X.shape[1]))
