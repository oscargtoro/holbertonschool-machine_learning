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

    t, m, _ = X.shape
    _, h = h_0.shape

    h_f = np.zeros((t + 1, m, h))
    h_b = np.zeros((t + 1, m, h))
    h_f[0] = h_0
    h_b[t] = h_t

    for step in range(t):
        h_f[step + 1] = bi_cell.forward(h_f[step], X[step])

    for step in range(t - 1, -1, -1):
        h_b[step] = bi_cell.backward(h_b[step + 1], X[step])

    H = np.concatenate((h_f[1:], h_b[0:t]), axis=-1)
    Y = bi_cell.output(H)

    return H, Y
