#!/usr/bin/env python3
"""Module for the class LSTMCell
"""

import numpy as np


class LSTMCell:
    """Represents an LSTM unit
    """

    def __init__(self, i, h, o):
        """Class constructor

        Args.
            i: dimensionality of the data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs
        """
        pass

    def forward(self, h_prev, c_prev, x_t):
        """Performs forward propagation for one time step

        Args.
            x_t: numpy.ndarray of shape (m, i) that contains the data input for
             the cell
                -m is the batche size for the data
            h_prev: numpy.ndarray of shape (m, h) containing the previous
            hidden state
            c_prev: numpy.ndarray of shape (m, h) containing the previous cell
            state

        Returns.
            The next hidden state, the next cell state and the output of the
            cell
        """
        pass
