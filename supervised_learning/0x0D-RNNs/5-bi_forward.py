#!/usr/bin/env python3
"""Module for the class BidirectionalCell
"""

import numpy as np


class BidirectionalCell:
    """Represents a bidirectional cell of an RNN
    """

    def __init__(self, i, h, o):
        """Class constructor

        Args.
            i: dimensionality of the data
            h: dimensionality of the hidden states
            o: dimensionality of the outputs
        """
        pass

    def forward(self, h_prev, x_t):
        """Calculates the hidden state in the forward direction for one time
        step

        Args.
            x_t: numpy.ndarray of shape (m, i) that contains the data input for
            the cell
                -m is the batch size for the data
            h_prev: numpy.ndarray of shape (m, h) containing the previous
            hidden state
        Returns:
            The next hidden state
        """
        pass
