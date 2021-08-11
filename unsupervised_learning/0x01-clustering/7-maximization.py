#!/usr/bin/env python3
"""Module for the function maximization
"""

import numpy as np


def maximization(X, g):
    """Calculates the maximization step in the EM algorithm for a GMM

    Args.
        X is a numpy.ndarray of shape (n, d) containing the data set
        g is a numpy.ndarray of shape (k, n) containing the posterior
        probabilities for each data point in each cluster

    Returns.
        A numpy.ndarray of shape (k,) containing the updated priors for each
        cluster, a numpy.ndarray of shape (k, d) containing the updated
        centroid means for each cluster and a numpy.ndarray of shape (k, d, d)
        containing the updated covariance matrices for each cluster,
        or None, None, None on failure
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None

    n, d = X.shape

    if not isinstance(g, np.ndarray) or len(g.shape) != 2 or g.shape[1] != n:
        return None, None, None

    pi = []
    m = []
    S = []
    for i in range(g.shape[0]):
        pi.append(np.sum(g[i]) / n)
        m.append(np.matmul(g[i], X) / np.sum(g[i]))
        S.append(np.matmul(g[i] * (X - m[i]).T, X - m[i]) / np.sum(g[i]))

    return np.array(pi), np.array(m), np.array(S)
