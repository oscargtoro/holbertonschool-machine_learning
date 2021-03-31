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
        C: a numpy.ndarray of shape (k, d) containing the centroid means for
        each cluster.
        clss: a numpy.ndarray of shape (n,) containing the index of the
        cluster in C that each data point belongs to.
        None on failure.
    """
    pass
