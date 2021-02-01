#!/usr/bin/env python3
"""Module for the function
convolve_grayscale_same(images, kernel).
"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """Performs a same convolution on grayscale images.

    Args:
        images: numpy.ndarray with shape (m, h, w) containing multiple
        grayscale images.
            - m: The number of images.
            - h: The height in pixels of the images.
            - w: The width in pixels of the images.
        kernel: numpy.ndarray with shape (kh, kw) containing the kernel
        for the convolution.
            - kh: The height of the kernel.
            - kw: The width of the kernel.

    Returns:
    A numpy.ndarray containing the convolved images.
    """

    in_d = images.shape[0]
    in_h = images.shape[1]
    in_w = images.shape[2]
    k_h = kernel.shape[0]
    k_w = kernel.shape[1]

    if k_h % 2 == 0:
        pad_h = k_h // 2
    else:
        pad_h = (k_h - 1) // 2
    if k_w % 2 == 0:
        pad_w = k_w // 2
    else:
        pad_w = (k_w - 1) // 2

    pad_size = ((0, 0), (pad_h, pad_h), (pad_w, pad_w))
    images_padded = np.pad(images,
                           pad_width=pad_size,
                           mode='constant',
                           constant_values=0)

    output = np.zeros((in_d, in_h, in_w))

    for h in range(in_h):
        for w in range(in_w):
            output[:, h, w] = (
                               kernel *
                               images_padded[:, h:h + k_h, w:w + k_w]
                               ).sum(axis=(1, 2))

    return output
