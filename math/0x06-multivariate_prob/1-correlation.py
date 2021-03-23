#!/usr/bin/env python3
"""Module for the function correlation
    correlation - alculates a correlation matrix
"""

import numpy as np


def correlation(C):
    """Calculates a correlation matrix

    Args.
        C is a numpy.ndarray of shape (d, d) containing a covariance matrix
            - d is the number of dimensions

    Returns.
        A numpy.ndarray of shape (d, d) containing the correlation matrix
    """

    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if len(C) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    diag = np.sqrt(np.diag(C))
    corr = np.outer(diag, diag)
    return C / corr
