#!/usr/bin/env python3
"""Module for the function inception_block that builds an inception block as
described in Going Deeper with Convolutions (2014).
"""

import tensorflow.keras as K


def inception_block(A_prev, filters):
    """Builds an inception block as described in Going Deeper with
    Convolutions (2014), ReLU is used in the incepction block.

    Args:
        A_prev: The output from the previous layer.
        filters: A tuple or list containing F1, F3R, F3,F5R, F5, FPP, in
        that order.
            -F1: The number of filters in the 1x1 convolution.
            -F3R: The number of filters in the 1x1 convolution before the 3x3
            convolution
            -F3: The number of filters in the 3x3 convolution.
            -F5R: The number of filters in the 1x1 convolution before the 5x5
            convolution.
            -F5: The number of filters in the 5x5 convolution.
            -FPP: The number of filters in the 1x1 convolution after the max
            pooling.

    Returns:
        The concatenated output of the inception block
    """
    pass
