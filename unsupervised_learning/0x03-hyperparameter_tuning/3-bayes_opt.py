#!/usr/bin/env python3
"""Module for the class BayesianOptimization
"""

import numpy as np

GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Performs Bayesian optimization on a noiseless 1D Gaussian process
    """
    def __init__(self,
                 f,
                 X_init,
                 Y_init,
                 bounds,
                 ac_samples,
                 l=1,
                 sigma_f=1,
                 xsi=0.01,
                 minimize=True):
        """Class constructor

        Args.
        f: the black-box function to be optimized
        X_init: a numpy.ndarray representing the inputs already sampled with
        the black-box function
        Y_init: a numpy.ndarray representing the outputs of the black-box
        function for each input in X_init
        bounds: a tuple of (min, max) representing the bounds of the space in
        which to look for the optimal point
        ac_samples: the number of samples that should be analyzed during
        acquisition
        l: the length parameter for the kernel
        sigma_f: the standard deviation given to the output of the black-box
        function
        xsi: the exploration-exploitation factor for acquisition
        minimize: a bool determining whether optimization should be performed
        for minimization (True) or maximization (False)
        """

        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], num=ac_samples)
        self.X_s = self.X_s.reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize
