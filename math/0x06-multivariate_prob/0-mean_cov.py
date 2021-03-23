#!/usr/bin/env python3
"""Module for the function mean_cov
    mean_cov - Calculates the mean and covariance of a data set
"""

import numpy as np


def mean_cov(X):
    """Calculates the mean and covariance of a data set

    Args.
        X: A numpy.ndarray of shape (n, d) containing the data set
            - n is the number of data points
            - d is the number of dimensions in each data point

    Returns:
        mean: A numpy.ndarray of shape (1, d) containing the mean of the data
        set
        cov: A numpy.ndarray of shape (d, d) containing the covariance matrix
        of the data set
    """

    mean = np.mean(X, axis=0).reshape(1, X.shape[1])
    cov = np.matmul(X.T - mean.T, X - mean) / X.shape[0]

    return mean, cov
