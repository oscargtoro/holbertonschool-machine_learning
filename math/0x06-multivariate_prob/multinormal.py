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
