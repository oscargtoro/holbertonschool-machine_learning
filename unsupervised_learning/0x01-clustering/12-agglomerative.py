#!/usr/bin/env python3
"""Module for the function agglomerative
"""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Performs agglomerative clustering on a dataset

    Args.
        X: is a numpy.ndarray of shape (n, d) containing the dataset
        dist: is the maximum cophenetic distance for all clusters

    Returns.
        A numpy.ndarray of shape (n,) containing the cluster indices for each
        data point.
    """

    Z = scipy.cluster.hierarchy.linkage(X, 'ward')
    clss = scipy.cluster.hierarchy.fcluster(Z, dist, criterion="distance")
    scipy.cluster.hierarchy.dendrogram(Z, color_threshold=dist)
    plt.show()

    return clss
