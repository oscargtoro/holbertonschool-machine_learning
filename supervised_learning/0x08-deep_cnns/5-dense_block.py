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
    pass
