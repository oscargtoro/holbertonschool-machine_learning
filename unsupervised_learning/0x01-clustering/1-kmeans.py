#!/usr/bin/env python3
"""Module for the function kmeans
"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """Performs K-means on a dataset

    Args.
        X: is a numpy.ndarray of shape (n, d) containing the dataset
            -n: is the number of data points
            -d: is the number of dimensions for each data point
        k: is a positive integer containing the number of clusters
        iterations: is a positive integer containing the maximum number of
        iterations that should be performed

    Returns:
        A numpy.ndarray of shape (k, d) containing the centroid means for
        each cluster and a numpy.ndarray of shape (n,) containing the index of
        the cluster in C that each data point belongs to, or None, None on
        failure.
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None

    low = X.min(axis=0)
    high = X.max(axis=0)

    c = np.random.uniform(low, high, (k, X.shape[1]))

    for _ in range(0, iterations):
        distances = np.sqrt(((X - c[:, np.newaxis])**2).sum(axis=2))
        minDistances = np.argmin(distances, axis=0)
        cNew = []
        for i in range(c.shape[0]):
            if i not in minDistances:
                cNew.append(np.random.uniform(low, high, (1, X.shape[1]))[0])
                continue
            cNew.append(X[np.argmin(distances, axis=0) == i].mean(axis=0))
        if np.array_equal(cNew, c):
            return np.array(cNew), minDistances
        else:
            c = np.array(cNew)

    return c, minDistances
