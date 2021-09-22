#!/usr/bin/env python3
"""Module for the class BayesianOptimization
"""

import numpy as np
from scipy.stats import norm

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

    def acquisition(self):
        """Calculates the next best sample location

        Returns.
            A numpy.ndarray representing the next best sample point and a
            numpy.ndarray containing the expected improvement of eachpotential
            sample
        """

        mu_s, sigma_s = self.gp.predict(self.X_s)

        if self.minimize is True:
            Y_s = np.min(self.gp.Y)
            imp = Y_s - mu_s - self.xsi

        else:
            Y_s = np.max(self.gp.Y)
            imp = mu_s - Y_s - self.xsi

        with np.errstate(divide='ignore'):
            Z = imp / sigma_s
            EI = imp * norm.cdf(Z) + sigma_s * norm.pdf(Z)
            EI[sigma_s == 0.0] = 0.0

        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI

    def optimize(self, iterations=100):
        """Optimizes the black-box function

        Args.
            iterations: the maximum number of iterations to perform

        Returns.
        A numpy.ndarray representing the optimal point and a numpy.ndarray
        representing the optimal function value
        """

        for _ in range(iterations):
            X_next, _ = self.acquisition()
            Y_next = self.f(X_next)

            if (X_next == self.gp.X).any():
                self.gp.X = self.gp.X[:-1]
                break

            self.gp.update(X_next, Y_next)

        if self.minimize:
            idx = np.argmin(self.gp.Y)

        else:
            idx = np.argmax(self.gp.Y)

        X_opt = self.gp.X[idx]
        Y_opt = self.gp.Y[idx]

        return X_opt, Y_opt
