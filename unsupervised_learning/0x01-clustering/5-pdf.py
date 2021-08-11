#!/usr/bin/env python3
"""Module for the function pdf
"""

import numpy as np


def pdf(X, m, S):
    """Calculates the probability density function of a Gaussian distribution

    Args.
        X: is a numpy.ndarray of shape (n, d) containing the data points whose
        PDF should be evaluated
        m: is a numpy.ndarray of shape (d,) containing the mean of the
        distribution
        S: is a numpy.ndarray of shape (d, d) containing the covariance of the
        distribution

    Returns.
        A numpy.ndarray of shape (n,) containing the PDF values for each data
        point or None on failure
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(X, np.ndarray) or len(S.shape) != 2:
        return None

    d = X.shape[1]

    if (d,) != m.shape:
        return None
    if (d, d) != S.shape:
        return None

    S_det = np.linalg.det(S)
    S_inv = np.linalg.inv(S)
    divisor = np.sqrt(((2 * np.pi) ** d) * S_det)

    exponent = (-0.5 * np.sum(np.matmul(S_inv,
                (X.T - m[:, np.newaxis])) *
                (X.T - m[:, np.newaxis]), axis=0))

    return np.maximum(np.exp(exponent) / divisor, 1e-300)
