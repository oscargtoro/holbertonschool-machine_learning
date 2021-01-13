#!/usr/bin/env python3
'''Module for normalize(X, m, s)
'''


def normalize(X, m, s):
    '''Normalizes (standardizes) a matrix.

    Args.
        X: numpy.ndarray of shape (d, nx) to normalize.
            - d: Number of data points.
            - nx: Number of features.
        m: numpy.ndarray of shape (nx,) that contains the mean of all features
           of X.
        s: numpy.ndarray of shape (nx,) that contains the standard deviation
           of all features of X.

    Returns.
        The normalized X matrix.
    '''

    return (X - m) / s
