#!/usr/bin/env python3
'''Module for the function
convolve_channels(images, kernel, padding='same', stride=(1, 1)).
'''

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    '''Performs a convolution on images with channels.

    Args:
        images: numpy.ndarray with shape (m, h, w, c) containing multiple
        grayscale images.
            - m: The number of images.
            - h: The height in pixels of the images.
            - w: The width in pixels of the images.
            - c: The number of channels in the image.
        kernel: numpy.ndarray with shape (kh, kw, c) containing the kernel
        for the convolution.
            - kh: The height of the kernel.
            - kw: The width of the kernel.
            - c: The number of channels in the kernel.
        padding: A tuple of (ph, pw), ‘same’, or ‘valid’.
            - ph: The padding for the height of the image.
            - pw: The padding for the width of the image.
        stride: A tuple of (sh, sw).
            - sh: The stride for the height of the image.
            - sw: The stride for the width of the image.

    Returns:
    A numpy.ndarray containing the convolved images.
    '''

    (in_d, in_h, in_w, in_c) = images.shape
    (k_h, k_w, k_c) = kernel.shape
    (s_h, s_w) = stride

    if padding is 'valid':
        p_h, p_w = 0, 0

    if padding is 'same':
        p_h = (((in_h - 1) * s_h + k_h - in_h) // 2) + 1
        p_w = (((in_w - 1) * s_w + k_w - in_w) // 2) + 1

    if isinstance(padding, tuple):
        p_h = padding[0]
        p_w = padding[1]

    pad_size = ((0, 0), (p_h, p_h), (p_w, p_w), (0, 0))
    out_h = ((in_h - k_h + 2 * p_h) // s_h) + 1
    out_w = ((in_w - k_w + 2 * p_w) // s_w) + 1
    output = np.zeros((in_d, out_h, out_w))

    images_padded = np.pad(images,
                           pad_width=pad_size,
                           mode='constant',
                           constant_values=0)

    for h in range(out_h):
        for w in range(out_w):
            output[:, h, w] = (
                               kernel *
                               images_padded[:,
                                             h * s_h:h * s_h + k_h,
                                             w * s_w:w * s_w + k_w,
                                             :]
                               ).sum(axis=(1, 2, 3))

    return output
