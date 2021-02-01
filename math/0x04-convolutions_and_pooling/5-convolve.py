#!/usr/bin/env python3
'''Module for the function
def convolve(images, kernels, padding='same', stride=(1, 1)):
'''

import numpy as np


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
        for the pooling.
            - kh: The height of the kernel.
            - kw: The width of the kernel.
            - c: The number of channels in the kernel.
            - nc: is the number of kernels.
        padding: A tuple of (ph, pw), ‘same’, or ‘valid’.
            - ph: The padding for the height of the image.
            - pw: The padding for the width of the image.
        stride: A tuple of (sh, sw).
            - sh: The stride for the height of the image.
            - sw: The stride for the width of the image.
        mode: Indicates the type of pooling ( max or avg).

    Returns:
    A numpy.ndarray containing the convolved images.
    '''

    (in_d, in_h, in_w, in_c) = images.shape
    (k_h, k_w, k_c, k_n) = kernels.shape
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
    output = np.zeros((in_d, out_h, out_w, k_n))

    images_padded = np.pad(images,
                           pad_width=pad_size,
                           mode='constant',
                           constant_values=0)

    for h in range(out_h):
        for w in range(out_w):
            for n in range(k_n):
                output[:, h, w, n] = (
                                      kernels[:, :, :, n] *
                                      images_padded[:,
                                                    h * s_h:h * s_h + k_h,
                                                    w * s_w:w * s_w + k_w,
                                                    :]
                                     ).sum(axis=(1, 2, 3))

    return output
