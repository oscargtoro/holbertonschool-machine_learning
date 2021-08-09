#!/usr/bin/env python3
"""Module for the class RNNCell
"""

import numpy as np


class RNNCell:
    """Represents a cell of a simple RNN
    """

    def __init__(self, i, h, o):
        """class constructor

        Args.
            i: the dimensionality of the data
            h: the dimensionality of the hidden state
            o: the dimensionality of the outputs
        """
        pass

    def forward(self, h_prev, x_t):
        """Performs forward propagation for one time step

        Args.
            x_t: numpy.ndarray of shape (m, i) that contains the data input for
            the cell
                -m is the batche size for the data
                -h_prev is a numpy.ndarray of shape (m, h) containing the
                previous hidden state

        Returns.
            The next hidden state and the output of the cell
        """
        pass