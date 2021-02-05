#!/usr/vin/env python3
'''Module for the function
pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max')
'''

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    '''Performs back propagation over a pooling layer of a neural network.

    Args:
        dA: numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
        partial derivatives with respect to the output of the pooling layer
            -m: The number of examples.
            -h_new: The height of the output.
            -w_new: The width of the output.
            -c: The number of channels.
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c) containing the
        output of the previous layer
            -h_prev: The height of the previous layer.
            -w_prev: The width of the previous layer.
        kernel_shape: A tuple of (kh, k_w) containing the size of the kernel
        for the pooling
            -kh: The kernel height.
            -kw: The kernel width.
        stride: tuple of (sh, sw) containing the strides for the pooling.
            -sh: The stride for the height.
            -sw: The stride for the width
        mode: string containing either max or avg, indicating whether to
        perform maximum or average pooling

    Returns:
        The partial derivatives with respect to the previous layer (dA_prev).
    '''

    (dA_m, dA_h, dA_w, dA_c) = dA.shape
    (k_h, k_w) = kernel_shape
    (s_h, s_w) = stride

    dA_prev = np.zeros_like(A_prev)

    for m in range(dA_m):
        for h in range(dA_h):
            for w in range(dA_w):
                for c in range(dA_c):
                    if mode is 'max':
                        Ap_slice = A_prev[m,
                                          h * s_h:h * s_h + k_h,
                                          w * s_w:w * s_w + k_w,
                                          c]
                        mask = np.where(Ap_slice == np.max(Ap_slice), 1, 0)
                        dA_prev[m,
                                h * s_h:h * s_h + k_h,
                                w * s_w:w * s_w + k_w,
                                c] += dA[m, h, w, c] * mask
                    if mode is 'avg':
                        avg = dA[m, h, w, c] / (k_w * k_h)
                        print(avg, dA[m, h, w, c]/(dA_h * dA_w))
                        break
                        dA_prev[m,
                                h * s_h:h * s_h + k_h,
                                w * s_w:w * s_w + k_w,
                                c] += np.ones(kernel_shape) * avg

    return dA_prev
