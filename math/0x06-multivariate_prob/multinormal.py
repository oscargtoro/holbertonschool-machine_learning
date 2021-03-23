#!/usr/bin/env python3
"""Module for the class MultiNormal
    MultiNormal - represents a Multivariate Normal distribution
"""

import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution
    """

    def __init__(self, data):
        """Class constructor

        Args.
            data is a numpy.ndarray of shape (d, n) containing the data set
                -n is the number of data points
                -d is the number of dimensions in each data point
        """

        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")
        mean = np.mean(data, axis=1, keepdims=True)
        cov = np.matmul(data - mean, data.T - mean.T) / (data.shape[1] - 1)
        self.mean = mean
        self.cov = cov

    def pdf(self, x):
        """Calculates the PDF at a data point

        Args.
            x is numpy.ndarray of shape (d, 1) containing the data point whose
            PDF should be calculated
                -d is the number of dimensions of the Multinomial instance

        Returns.
            The value of the PDF
        """

        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        if x.shape != (self.cov.shape[0], 1):
            e_msg = "x must have the shape ({}, 1)".format(self.cov.shape[0])
            raise ValueError(e_msg)

        denominator = np.sqrt(((2 * np.pi) ** x.shape[0])
                              * np.linalg.det(self.cov))
        exponent = -0.5 * np.matmul(np.matmul((x - self.mean).T,
                                    np.linalg.inv(self.cov)), x - self.mean)

        return (1 / denominator) * np.exp(exponent[0][0])
