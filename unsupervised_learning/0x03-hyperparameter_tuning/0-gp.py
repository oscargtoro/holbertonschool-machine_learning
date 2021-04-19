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

        pass

    def kernel(self, X1, X2):
        """Calculates the covariance kernel matrix between two matrices

        Args.
            X1: a numpy.ndarray of shape (m, 1)
            X2: a numpy.ndarray of shape (n, 1)
        Returns:
            The covariance kernel matrix as a numpy.ndarray
        """

        pass
