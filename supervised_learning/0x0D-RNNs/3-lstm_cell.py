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

        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, h))

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

        con = np.concatenate((h_prev, x_t), axis=1)
        f_t = 1 / (1 + np.exp(-(np.matmul(con, self.Wf) + self.bf)))
        u_t = 1 / (1 + np.exp(-(np.matmul(con, self.Wu) + self.bu)))
        o_t = 1 / (1 + np.exp(-(np.matmul(con, self.Wo) + self.bo)))
        c_t = np.tanh(np.matmul(con, self.Wc) + self.bc)
        c_next = f_t * c_prev + u_t * c_t
        h_next = o_t * np.tanh(c_next)
        sof_mm = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(sof_mm) / np.sum(np.exp(sof_mm), axis=1, keepdims=True)

        return h_next, c_next, y
