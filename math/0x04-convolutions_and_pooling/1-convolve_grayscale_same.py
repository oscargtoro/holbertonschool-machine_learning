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
    out_h = in_h
    out_w = in_w
    out_d = in_d

    # Calculate the number of zeros which are needed to add as padding
    pad_along_height = max((out_h - 1) + k_h - in_h, 0)
    pad_along_width = max((out_w - 1) + k_w - in_w, 0)
    # amount of zero padding on the top
    pad_top = pad_along_height // 2
    # amount of zero padding on the bottom
    pad_bottom = pad_along_height - pad_top
    # amount of zero padding on the left
    pad_left = pad_along_width // 2
    # amount of zero padding on the right
    pad_right = pad_along_width - pad_left
    output = np.zeros((out_d, out_h, out_w))

    images_padded = np.zeros((in_d,
                              in_h + pad_along_height,
                              in_w + pad_along_width))
    images_padded[:, pad_top:-pad_bottom, pad_left:-pad_right] = images

    for h in range(out_h):
        for w in range(out_w):
            output[:, h, w] = (
                               kernel *
                               images_padded[:, h:h + k_h, w:w + k_w]
                               ).sum(axis=(1, 2))

    return output
