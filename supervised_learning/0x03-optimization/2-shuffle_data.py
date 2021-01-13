#!/usr/bin/env python3
'''Module for shuffle_data(X, Y).
'''

import numpy as np


def shuffle_data(X, Y):
    '''Shuffles the data points in two matrices the same way.

    Args.
        X: First numpy.ndarray of shape (m, nx) to shuffle.
            - m: Number of data points.
            - nx: Number of features in X.
        Y: Second numpy.ndarray of shape (m, ny) to shuffle.
            - m: Same number of data points as in X.
            - ny: Number of features in Y.

    Returns.
        The shuffled X and Y matrices.
    '''
    rnd = np.random.permutation(X.shape[0])
    return X[rnd], Y[rnd]
