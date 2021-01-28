#!/usr/bin/env python3
'''Module for the function
convolve(images, kernels, padding='same', stride=(1, 1)).
'''

import numpy as np
from math import ceil, floor


def convolve(images, kernels, padding='same', stride=(1, 1)):
    '''Performs a convolution on images using multiple kernels.

    Args:
        images: numpy.ndarray with shape (m, h, w, c) containing multiple
        images.
            - m: The number of images.
            - h: The height in pixels of the images.
            - w: The width in pixels of the images.
            - c: The number of channels in the image.
        kernel: numpy.ndarray with shape (kh, kw, c, nc) containing the kernel
        for the convolution.
            - kh: The height of the kernel.
            - kw: The width of the kernel.
            - nc: The number of kernels.
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
