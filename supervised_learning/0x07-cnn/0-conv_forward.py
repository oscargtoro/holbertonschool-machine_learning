#!/usr/bin/env python3
'''Module for the function
conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)).
'''

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    '''Performs forward propagation over a convolutional layer of a neural
    network.

    Args:
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
        the output of the previous layer.
            - m: The number of examples.
            - h_prev: The height of the previous layer.
            - w_prev: The width of the previous layer.
            - c_prev: The number of channels in the previous layer.
        W: numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the
        kernels for the convolution.
            - kh: The filter height.
            - kw: The filter width.
            - c_prev: The number of channels in the previous layer.
            - c_new: The number of channels in the output.
        b: numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
        applied to the convolution.
        activation: Activation function applied to the convolution.
        padding: A string that is either same or valid, indicating the type of
        padding used.
        stride: A tuple of (sh, sw) containing the strides for the convolution.
            - sh: The stride for the height.
            - sw: The stride for the width.

    Returns:
    The output of the convolutional layer.
    '''
