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

    # define top, bottom, left and right padding
    if k_h % 2 == 0:
        pad_h = k_h
    else:
        pad_h = k_h - 1
    if k_w % 2 == 0:
        pad_w = k_w
    else:
        pad_w = k_w - 1
    pad_t = pad_h // 2
    pad_b = pad_h - pad_t
    pad_l = pad_w // 2
    pad_r = pad_w - pad_l

    # create a new numpy array with the necessary shape for same output
    images_padded = np.zeros((in_d, in_h + pad_h, in_w + pad_w))
    images_padded[:, pad_t:-pad_b, pad_l:-pad_r] = images

    # create the output numpy array
    out_h = in_h + (2 * (pad_t)) - k_h + 1
    out_w = in_w + (2 * (pad_l)) - k_w + 1
    output = np.zeros((in_d, out_h, out_w))

    for h in range(out_h):
        for w in range(out_w):
            output[:, h, w] = (
                               kernel *
                               images_padded[:, h:h + k_h, w:w + k_w]
                               ).sum(axis=(1, 2))

    return output
