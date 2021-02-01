#!/usr/bin/env python3
'''Module for the function
def pool(images, kernel_shape, stride, mode='max').
'''

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    '''performs pooling on images.

    Args:
        images: numpy.ndarray with shape (m, h, w, c) containing multiple
        images.
            - m: The number of images.
            - h: The height in pixels of the images.
            - w: The width in pixels of the images.
            - c: The number of channels in the image.
        kernel: numpy.ndarray with shape (kh, kw) containing the kernel
        for the convolution.
            - kh: The height of the kernel.
            - kw: The width of the kernel.
        stride: A tuple of (sh, sw).
            - sh: The stride for the height of the image.
            - sw: The stride for the width of the image.
        mode: Indicates the type of pooling.
            - max: Indicates max pooling
            - avg: Indicates average pooling

    Returns:
    A numpy.ndarray containing the pooled images.
    '''

    (in_d, in_h, in_w, in_c) = images.shape
    (k_h, k_w) = kernel_shape
    (s_h, s_w) = stride

    out_h = ((in_h - k_h) // s_h) + 1
    out_w = ((in_w - k_w) // s_w) + 1
    output = np.zeros((in_d, out_h, out_w, in_c))

    for h in range(out_h):
        for w in range(out_w):
            if mode is 'avg':
                output[:, h, w, :] = np.average(images[:,
                                                       h * s_h:h * s_h + k_h,
                                                       w * s_w:w * s_w + k_w,
                                                       :], axis=(1, 2))
            if mode is 'max':
                output[:, h, w, :] = images[:,
                                            h * s_h:h * s_h + k_h,
                                            w * s_w:w * s_w + k_w,
                                            :].max(axis=(1, 2))

    return output
