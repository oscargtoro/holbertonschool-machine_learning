#!/usr/bin/env python3
'''Module for the function
convolve_grayscale(images, kernel, padding='same', stride=(1, 1)).
'''

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    '''Performs a convolution on grayscale images.

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
        padding: A tuple of (ph, pw), ‘same’, or ‘valid’.
            - ph: The padding for the height of the image.
            - pw: The padding for the width of the image.
        stride: A tuple of (sh, sw).
            - sh: The stride for the height of the image.
            - sw: The stride for the width of the image.

    Returns:
    A numpy.ndarray containing the convolved images.
    '''
    pass
