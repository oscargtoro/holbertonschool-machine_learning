#!/usr/bin/env python3
'''Module for the function
pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max').
'''

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    '''Performs forward propagation over a pooling layer of a neural network.

    Args:
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
        the output of the previous layer.
            - m: The number of examples.
            - h_prev: The height of the previous layer.
            - w_prev: The width of the previous layer.
            - c_prev: The number of channels in the previous layer.
        kernel_shape: A tuple of (kh, kw) containing the size of the kernel
        for the pooling
            - kh: The kernel height.
            - kw: The kernel width.
        stride: A tuple of (sh, sw) containing the strides for the pooling.
            - sh: The stride for the height.
            - sw: The stride for the width.
        mode: A string containing either max or avg, indicating whether to
        perform maximum or average pooling.

    Returns:
        The output of the pooling layer
    '''

    (in_d, in_h, in_w, in_c) = A_prev.shape
    (k_h, k_w) = kernel_shape
    (s_h, s_w) = stride

    out_h = ((in_h - k_h) // s_h) + 1
    out_w = ((in_w - k_w) // s_w) + 1

    output = np.zeros((in_d, out_h, out_w, in_c))

    for h in range(out_h):
        for w in range(out_w):
            out_d = A_prev[:, h * s_h:h * s_h + k_h, w * s_w:w * s_w + k_w, :]
            if mode is 'max':
                output[:, h, w, :] = out_d.max(axis=(1, 2))

            if mode is 'avg':
                output[:, h, w, :] = np.average(out_d, axis=(1, 2))

    return output
