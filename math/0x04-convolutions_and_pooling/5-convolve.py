#!/usr/bin/env python3
'''Module for the function
pool(images, kernel_shape, stride, mode='max').
'''

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    '''Performs pooling on images.

    Args:
        images: numpy.ndarray with shape (m, h, w, c) containing multiple
        images.
            - m: The number of images.
            - h: The height in pixels of the images.
            - w: The width in pixels of the images.
            - c: The number of channels in the image.
        kernel: numpy.ndarray with shape (kh, kw, c, nc) containing the kernel
        for the pooling.
            - kh: The height of the kernel.
            - kw: The width of the kernel.
        stride: A tuple of (sh, sw).
            - sh: The stride for the height of the image.
            - sw: The stride for the width of the image.
        mode: Indicates the type of pooling ( max or avg).

    Returns:
    A numpy.ndarray containing the convolved images.
    '''
    pass
