#!/usr/bin/env python3
"""Module for the method rnn
"""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """Performs forward propagation for a simple RNN

    Args.
        rnn_cell: instance of RNNCell that will be used for the forward
        propagation
        X: data to be used, given as a numpy.ndarray of shape (t, m, i)
            -t is the maximum number of time steps
            -m is the batch size
            -i is the dimensionality of the data
        h_0: initial hidden state, given as a numpy.ndarray of shape (m, h)
            -m is the batch size
            -h is the dimensionality of the hidden state

    Returns: H, Y
        A numpy.ndarray containing all of the hidden states and a numpy.ndarray
         containing all of the outputs
    """
    pass
