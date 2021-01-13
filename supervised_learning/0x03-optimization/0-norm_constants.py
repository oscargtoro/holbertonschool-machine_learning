#!/usr/bin/env python3
'''Module for normalization_constants(X)
'''
import numpy as np


def normalization_constants(X):
    '''Calculates the normalization (standardization) constants of a matrix.

    Args.
        X: numpy.ndarray of shape (m, nx) to normalize.
            - m: Number of data points.
            - nx: Number of features.
    Returns.
        The mean and standard deviation of each feature, respectively.
    '''

    return np.mean(X, axis=0), np.std(X, axis=0)
