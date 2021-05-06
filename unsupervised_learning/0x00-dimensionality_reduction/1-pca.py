#!/usr/bin/env python3
"""Module for the function pca
    pca: performs PCA on a dataset
"""

import numpy as np


def pca(X, ndim):
    """Performs PCA on a dataset

    Args.
        X: is a numpy.ndarray of shape (n, d) where:
            -n is the number of data points
            -d is the number of dimensions in each point
        ndim: is the new dimensionality of the transformed X

    Returns:
        T, a numpy.ndarray of shape (n, ndim) containing the transformed
        version of X
    """

    X = X - np.mean(X, axis=0)
    _, s, vh = np.linalg.svd(X)
    T = np.matmul(X, vh[:ndim].T)
    return T
