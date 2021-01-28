#!/usr/bin/env python3
'''Module for the function
convolve_grayscale_valid(images, kernel).
'''

import numpy as np


def convolve_grayscale_valid(images, kernel):
    '''Performs a valid convolution on grayscale images.

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
    '''

    output_nimgs = images.shape[0]
    output_height = images.shape[1] - kernel.shape[0] + 1
    output_width = images.shape[2] - kernel.shape[1] + 1
    k_h = kernel.shape[0]
    k_w = kernel.shape[1]
    output = np.zeros((output_nimgs, output_height, output_width))

    for h in range(output_height):
        for w in range(output_width):
            output[:, h, w] = (kernel * images[:,
                                               h:h + k_h,
                                               w:w + k_w]).sum(axis=(1, 2))

    return output
