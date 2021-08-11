#!/usr/bin/env python3
"""Module for the function gmm
"""

import sklearn.mixture


def gmm(X, k):
    """Calculates a GMM from a dataset

    Args.
        X is a numpy.ndarray of shape (n, d) containing the dataset
        k is the number of clusters

    Returns. pi, m, S, clss, bic
        A numpy.ndarray of shape (k,) containing the cluster priors,
        a numpy.ndarray of shape (k, d) containing the centroid means,
        a numpy.ndarray of shape (k, d, d) containing the covariance matrices,
        a numpy.ndarray of shape (n,) containing the cluster indices for each
        data point and a numpy.ndarray of shape (kmax - kmin + 1) containing
        the BIC value for each cluster size tested.
    """

    _gmm = sklearn.mixture.GaussianMixture(n_components=k).fit(X)
    pi = _gmm.weights_
    m = _gmm.means_
    S = _gmm.covariances_
    clss = _gmm.predict(X)
    bic = _gmm.bic(X)

    return pi, m, S, clss, bic
