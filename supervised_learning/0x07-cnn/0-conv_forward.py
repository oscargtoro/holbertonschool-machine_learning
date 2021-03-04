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

    in_d, in_h, in_w, in_c = A_prev.shape
    W_h, W_w, _, W_nc = W.shape
    s_h, s_w = stride
    p_h = ((in_h * (s_h - 1)) - s_h + W_h) // 2
    p_w = ((in_w * (s_w - 1)) - s_w + W_w) // 2

    if padding == 'valid':
        p_h = 0
        p_w = 0

    out_h = ((in_h - W_h + 2 * p_h) // s_h) + 1
    out_w = ((in_w - W_w + 2 * p_w) // s_w) + 1

    pad_size = ((0, 0), (p_h, p_h), (p_w, p_w), (0, 0))
    output = np.zeros((in_d, out_h, out_w, W_nc))

    imgs_padded = np.pad(A_prev,
                         pad_width=pad_size,
                         mode='constant',
                         constant_values=0)

    for h in range(out_h):
        for w in range(out_w):
            for n in range(W_nc):
                output[:, h, w, n] = (
                                      W[:, :, :, n] *
                                      imgs_padded[:,
                                                  h * s_h:h * s_h + W_h,
                                                  w * s_w:w * s_w + W_w,
                                                  :]
                                      ).sum(axis=(1, 2, 3))

    return activation(output + b)
