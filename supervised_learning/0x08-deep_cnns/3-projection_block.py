#!/usr/bin/env python3
"""Module for the function projection_block that builds a projection block as
described in Deep Residual Learning for Image Recognition (2015).
https://arxiv.org/pdf/1512.03385.pdf
"""

import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """Builds a projection block as described in Deep Residual Learning for
    Image Recognition (2015).

    Args:
        A_prev: The output from the previous layer.
        filters: A tuple or list containing F11, F3, F12, in that order.
            -F11: The number of filters in the first 1x1 convolution.
            -F3: The number of filters in the 3x3 convolution.
            -F12: The number of filters in the second 1x1 convolution as well
            as the 1x1 convolution in the shortcut connection.
        s: The stride of the first convolution in both the main path and the
        shortcut connection.

    Returns:
        The activated output of the projection block.
    """
    pass
