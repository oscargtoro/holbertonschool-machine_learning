#!/usr/bin/env python3
"""Module for the function expectation
"""

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Calculates the expectation step in the EM algorithm for a GMM

    Args.
        X: is a numpy.ndarray of shape (n, d) containing the data set
        pi: is a numpy.ndarray of shape (k,) containing the priors for each
        cluster
        m: is a numpy.ndarray of shape (k, d) containing the centroid means
        for each cluster
        S: is a numpy.ndarray of shape (k, d, d) containing the covariance
        matrices for each cluster

    Returns:
        A numpy.ndarray of shape (k, n) containing the posterior probabilities
        for each data point in each cluster and the total log likelihood, or
        None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if (k, d) != m.shape:
        return None, None
    if (k, d, d) != S.shape:
        return None, None

    tmp = np.zeros((k, n))

    for i in range(k):
        P = pdf(X, m[i], S[i])
        tmp[i, :] = P * pi[i]

    posProbabilities = tmp / np.sum(tmp, axis=0)
    logLikelihood = np.sum(np.log(np.sum(tmp, axis=0)))

    return posProbabilities, logLikelihood
