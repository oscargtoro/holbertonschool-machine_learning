#!/usr/bin/env python3
"""Module for the method deep_rnn
"""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Performs forward propagation for a deep RNN

    Args.
        rnn_cells: list of RNNCell instances of length l that will be used for
        the forward propagation
            -l is the number of layers
        X: data to be used, given as a numpy.ndarray of shape (t, m, i)
            -t is the maximum number of time steps
            -m is the batch size
            -i is the dimensionality of the data
        h_0: initial hidden state, given as a numpy.ndarray of shape (l, m, h)
            -h is the dimensionality of the hidden state
    Returns: H, Y
        A numpy.ndarray containing all of the hidden states and a numpy.ndarray
        containing all of the outputs
    """

    t, m, _ = X.shape
    l, _, h = h_0.shape

    H = np.zeros((t + 1, l, m, h))

    H[0, :, :, :] = h_0

    Y = []

    for step in range(t):
        for layer in range(l):
            if layer == 0:
                h_next, y = rnn_cells[layer].forward(H[step, layer], X[step])
            else:
                h_next, y = rnn_cells[layer].forward(H[step, layer], h_next)
            H[step + 1, layer] = h_next
        Y.append(y)

    return H, np.array(Y)
