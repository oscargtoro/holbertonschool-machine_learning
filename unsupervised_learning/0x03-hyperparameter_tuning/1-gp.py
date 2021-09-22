#!/usr/bin/env python3
"""Module for the class GaussianProcess
"""

import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian process
    """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Contructor definition

        Args.
            X_init: a numpy.ndarray representing the inputs already sampled
            with the black-box function
            Y_init: a numpy.ndarray representing the outputs of the black-box
            function for each input in X_init
            l: the length parameter for the kernel
            sigma_f: the standard deviation given to the output of the
            black-box function
        """

        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """Calculates the covariance kernel matrix between two matrices

        Args.
            X1: a numpy.ndarray of shape (m, 1)
            X2: a numpy.ndarray of shape (n, 1)
        Returns:
            The covariance kernel matrix as a numpy.ndarray
        """

        sqdist = np.sum(X1**2, 1).reshape(-1, 1) \
            + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)

    def predict(self, X_s):
        """Predicts the mean and standard deviation of points in a Gaussian
        process
            X_s: a numpy.ndarray containing all of the points whose mean and
            standard deviation should be calculated

        Returns:
            A numpy.ndarray containing the mean for each point in X_s,
            respectively and a numpy.ndarray containing the variance for each
            point in X_s
        """

        K = self.K
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(K)
        mu_s = K_s.T.dot(K_inv).dot(self.Y)
        mu_s = mu_s.reshape(-1)
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
        var_s = np.diag(cov_s)

        return mu_s, var_s
