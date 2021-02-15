#!/usr/bin/env python3
"""Module for the function dense_block that builds a dense block as described
in Densely Connected Convolutional Networks.
https://arxiv.org/pdf/1608.06993.pdf
"""

import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Builds a dense block as described in Densely Connected Convolutional
    Networks.

    Args:
        X: The output from the previous layer.
        nb_filters: An integer representing the number of filters in X.
        growth_rate: The growth rate for the dense block.
        layers: The number of layers in the dense block.

    Returns:
        The concatenated output of each layer within the Dense Block and the
        number of filters within the concatenated outputs, respectively.
    """

    init = K.initializers.he_normal()
    output = [X]
    # don't understand why, just that the output expected is 4 times the
    # growth rate
    c11f = growth_rate * 4

    # for each layer in layers we use bottleneck layers (1x1 layers) before
    # the 3x3 conv
    for layer in range(layers):
        X_prev = output[layer]
        H = K.layers.BatchNormalization()(X_prev)
        H = K.layers.Activation(activation="relu")(H)
        # bottleneck layer
        X = K.layers.Conv2D(filters=c11f,
                            kernel_size=1,
                            padding="same",
                            kernel_initializer=init)(H)
        H = K.layers.BatchNormalization()(X)
        H = K.layers.Activation(activation="relu")(H)
        X = K.layers.Conv2D(filters=growth_rate,
                            kernel_size=3,
                            padding="same",
                            kernel_initializer=init)(H)
        nb_filters += growth_rate
        output.append(K.layers.Concatenate()([output[layer], X]))

    return output, nb_filters
