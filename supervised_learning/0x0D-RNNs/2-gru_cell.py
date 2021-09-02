#!/usr/bin/env python3
"""Module for the class GRUCell
"""

import numpy as np


class GRUCell:
    """Represents a gated recurrent unit
    """

    def __init__(self, i, h, o):
        """class constructor

        Args.
            i: dimensionality of the data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs
        """

        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Performs forward propagation for one time step

        Args.
            x_t: numpy.ndarray of shape (m, i) that contains the data input for
            the cell
                -m is the batche size for the data
            h_prev: numpy.ndarray of shape (m, h) containing the previous
            hidden state

        Returns:
            The next hidden state and the output of the cell
        """

        con = np.concatenate((h_prev, x_t), axis=1)
        z_t = 1 / (1 + np.exp(np.matmul(con, self.Wz) + self.bz))
        r_t = 1 / (1 + np.exp(np.matmul(con, self.Wr) + self.br))
        imd = np.concatenate((r_t * h_prev, x_t), axis=1)
        h_t = np.tanh(np.matmul(imd, self.Wh) + self.bh)
        h_next = (1 - z_t) * h_prev + z_t *h_t
        sof_con = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(sof_con) / np.sum(np.exp(sof_con), axis=1, keepdims=True)

        return h_next, y
