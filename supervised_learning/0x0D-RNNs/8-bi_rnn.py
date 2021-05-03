#!/usr/bin/env python3
"""Module for the method bi_rnn
"""

import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """Performs forward propagation for a bidirectional RNN

    Args.
        bi_cell: an instance of BidirectinalCell that will be used for the
        forward propagation
        X: data to be used, given as a numpy.ndarray of shape (t, m, i)
            -t is the maximum number of time steps
            -m is the batch size
            -i is the dimensionality of the data
        h_0: initial hidden state in the forward direction, given as a
        numpy.ndarray of shape (m, h)
            -h is the dimensionality of the hidden state
        h_t: initial hidden state in the backward direction, given as a
        numpy.ndarray of shape (m, h)

    Returns.
        a numpy.ndarray containing all of the concatenated hidden states and a
        numpy.ndarray containing all of the outputs
    """
    pass
