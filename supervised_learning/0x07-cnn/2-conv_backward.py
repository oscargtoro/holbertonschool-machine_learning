#!/usr/bin/env python3
'''Module for the function
conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)).
'''

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    '''Performs back propagation over a convolutional layer of a neural
    network.

    Args:
        dZ: numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
        partial derivatives with respect to the unactivated output of the
        convolutional layer.
            - m: The number of examples.
            - h_new: The height of the output.
            - w_new: The width of the output.
            - c_new: The number of channels in the output.
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
        the output of the previous layer.
            - h_prev: The height of the previous layer.
            - w_prev: The width of the previous layer.
            - c_prev: The number of channels in the previous layer.
        W: numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the
        kernels for the convolution.
            - kh: The filter height.
            - kw: The filter width.
        b: numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
        applied to the convolution.
        padding: A string that is either same or valid, indicating the type of
        padding used.
        stride: tuple of (sh, sw) containing the strides for the convolution.
            - sh: The stride for the height.
            - sw: The stride for the width.

        Returns:
            The partial derivatives with respect to the previous
            layer (dA_prev), the kernels (dW), and the biases (db),
            respectively.
    '''

    (dZ_m, dZ_h, dZ_w, dZ_c) = dZ.shape
    (in_m, in_h, in_w, in_c) = A_prev.shape
    (W_h, W_w, W_cp, W_cn) = W.shape
    (s_h, s_w) = stride

    if padding is 'valid':
        p_h, p_w = 0, 0

    if padding is 'same':
        p_h = (((in_h - 1) * s_h + W_h - in_h) // 2) + 1
        p_w = (((in_w - 1) * s_w + W_w - in_w) // 2) + 1

    pad_dim = ((0,), (p_h,), (p_w,), (0,))

    A_padded = np.pad(A_prev,
                      pad_width=pad_dim,
                      mode='constant',
                      constant_values=0)
    dA_prev = np.zeros(A_padded.shape)
    dW = np.zeros((W_h, W_w, W_cp, W_cn))
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for m in range(dZ_m):
        for h in range(dZ_h):
            for w in range(dZ_w):
                for c in range(dZ_c):
                    dA_prev[m,
                            h * s_h:h * s_h + W_h,
                            w * s_w:w * s_w + W_w,
                            :] += W[:, :, :, c] * dZ[m, h, w, c]
                    dW[:, :, :, c] += A_padded[m,
                                               h * s_h:h * s_h + W_h,
                                               w * s_w:w * s_w + W_w,
                                               :] * dZ[m, h, w, c]

    if padding is 'same':
        dA_prev = dA_prev[:, p_h:-p_h, p_w:-p_w, :]

    return dA_prev, dW, db
