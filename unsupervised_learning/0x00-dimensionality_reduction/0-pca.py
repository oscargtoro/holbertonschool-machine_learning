#!/usr/bin/env python3
"""Module for the function pca
    pca: performs PCA on a dataset
"""

import numpy as np


def pca(X, var=0.95):
    """Performs PCA on a dataset

    Args.
        X: is a numpy.ndarray of shape (n, d) where:
            -n is the number of data points
            -d is the number of dimensions in each point
        var: is the fraction of the variance that the PCA transformation
        should maintain

    Returns:
        The weights matrix, W, that maintains var fraction of X‘s original
        variance
    """

    pass
