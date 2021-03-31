#!/usr/bin/env python3
"""Module for the function kmeans
"""

import sklearn.cluster


def kmeans(X, k):
    """Performs K-means on a dataset

    Args.
        X: is a numpy.ndarray of shape (n, d) containing the dataset
        k: is the number of clusters
    Returns. C, clss
        A numpy.ndarray of shape (k, d) containing the centroid means for each
        cluster and a numpy.ndarray of shape (n,) containing the index of the
        cluster in C that each data point belongs to.
    """
    pass
