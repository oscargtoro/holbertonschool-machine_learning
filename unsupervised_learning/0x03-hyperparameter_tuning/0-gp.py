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
