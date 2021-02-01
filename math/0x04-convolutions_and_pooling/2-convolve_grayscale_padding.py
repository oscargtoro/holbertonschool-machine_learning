#!/usr/bin/env python3
'''Module for the function
convolve_grayscale_padding(images, kernel, padding).
'''

import numpy as np


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
    in_d = images.shape[0]
    in_h = images.shape[1]
    in_w = images.shape[2]
    k_h = kernel.shape[0]
    k_w = kernel.shape[1]

    pad_size = ((0, 0), (padding[0], padding[0]), (padding[1], padding[1]))
    images_padded = np.pad(images,
                           pad_width=pad_size,
                           mode='constant',
                           constant_values=0)

    out_h = in_h + (2 * (padding[0])) - k_h + 1
    out_w = in_w + (2 * (padding[1])) - k_w + 1
    output = np.zeros((in_d, out_h, out_w))

    print(images.shape)
    print(images_padded.shape)
    print(output.shape)

    for h in range(out_h):
        for w in range(out_w):
            output[:, h, w] = (
                               kernel *
                               images_padded[:, h:h + k_h, w:w + k_w]
                               ).sum(axis=(1, 2))

    return output
