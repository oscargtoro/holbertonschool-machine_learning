#!/usr/bin/env python3
'''Module for the function
convolve_grayscale_padding(images, kernel, padding).
'''

import numpy as np
from math import ceil, floor


def convolve_grayscale_padding(images, kernel, padding):
    '''Performs a convolution on grayscale images with custom padding.

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
        padding: A tuple of (ph, pw).
            - ph: The padding for the height of the image.
            - pw: The padding for the width of the image.

    Returns:
    A numpy.ndarray containing the convolved images.
    '''
    pass
