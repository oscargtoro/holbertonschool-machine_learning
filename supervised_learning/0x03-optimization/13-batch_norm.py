#!/usr/bin/env python3
"""Module for the function batch_norm
"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Normalizes an unactivated output of a neural network using batch
    normalization

    Args.
        Z: is a numpy.ndarray of shape (m, n) that should be normalized
        m: is the number of data points
        n: is the number of features in Z
        gamma: is a numpy.ndarray of shape (1, n) containing the scales used
        for batch normalization
        beta: is a numpy.ndarray of shape (1, n) containing the offsets used
        for batch normalization
        epsilon: is a small number used to avoid division by zero
    Returns:
        The normalized Z matrix
    """

    m = Z.shape[0]
    u = (1 / m) * np.sum(Z, axis=0)
    sigma = ((1 / m) * np.sum((Z - u) ** 2, axis=0))
    z_norm = (Z - u) / ((sigma + epsilon) ** 0.5)
    return gamma * z_norm + beta
