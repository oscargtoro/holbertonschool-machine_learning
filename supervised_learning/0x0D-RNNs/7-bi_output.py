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

        self.Whf = np.random.randn(i + h, h)
        self.Whb = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h + h, o)
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

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

        con = np.concatenate((h_prev, x_t), axis=1)

        return np.tanh(np.matmul(con, self.Whf) + self.bhf)

    def backward(self, h_next, x_t):
        """Calculates the hidden state in the backward direction for one time
        step

        Args.
            x_t: numpy.ndarray of shape (m, i) that contains the data input for
            the cell
                -m is the batch size for the data
            h_next: numpy.ndarray of shape (m, h) containing the next hidden
            state

        Returns.
            The previous hidden state
        """

        con = np.concatenate((h_next, x_t), axis=1)

        return np.tanh(np.matmul(con, self.Whb) + self.bhb)

    def output(self, H):
        """That calculates all outputs for the RNN

        Args.
            H: numpy.ndarray of shape (t, m, 2 * h) that contains the
            concatenated hidden states from both directions, excluding their
            initialized states
                -t is the number of time steps
                -m is the batch size for the data
                -h is the dimensionality of the hidden states
        Returns.
            The outputs
        """
        t, _, _ = H.shape

        Y = []

        for step in range(t):
            sof = np.matmul(H[step], self.Wy) + self.by
            Y.append(np.exp(sof) / np.sum(np.exp(sof), axis=1, keepdims=True))

        return np.array(Y)
