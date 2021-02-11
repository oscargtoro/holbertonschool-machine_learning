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

    F1, F3R, F3, F5R, F5, FPP = filters
    concatf = []

    # Perform 1x1 convolution
    concatf.append(K.layers.Conv2D(filters=F1,
                                   kernel_size=1,
                                   padding="same",
                                   activation="relu")(A_prev))

    # 1x1 convolution for next 3x3 convolution
    # improving performance and accuracy
    F3R_layer = K.layers.Conv2D(filters=F3R,
                                kernel_size=1,
                                padding="same",
                                activation="relu")(A_prev)

    # 3x3 convolution using output from 1x1 convolution
    concatf.append(K.layers.Conv2D(filters=F3,
                                   kernel_size=3,
                                   padding="same",
                                   activation="relu")(F3R_layer))

    # 1x1 convolution for next 5x5 convolution
    # improving performance and accuracy
    F5R_layer = K.layers.Conv2D(filters=F5R,
                                kernel_size=1,
                                padding="same",
                                activation="relu")(A_prev)

    # 3x3 convolution using output from 1x1 convolution
    concatf.append(K.layers.Conv2D(filters=F5,
                                   kernel_size=5,
                                   padding="same",
                                   activation="relu")(F5R_layer))

    # Perform max pooling in the previous output
    maxp_layer = K.layers.MaxPool2D(pool_size=3,
                                    strides=1,
                                    padding="same")(A_prev)

    # 1x1 convolution for next 5x5 convolution
    # improving performance and accuracy
    concatf.append(K.layers.Conv2D(filters=FPP,
                                   kernel_size=1,
                                   padding="same",
                                   activation="relu")(maxp_layer))

    return K.layers.Concatenate(axis=-1)(concatf)
