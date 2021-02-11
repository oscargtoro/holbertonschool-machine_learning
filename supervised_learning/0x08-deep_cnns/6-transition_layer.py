#!/usr/bin/env python3
"""module for the function transition_layer that builds a transition layer as
described in Densely Connected Convolutional Networks.
https://arxiv.org/pdf/1608.06993.pdf
"""

import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """Builds a transition layer as described in Densely Connected
    Convolutional Networks.

    Args:
        X: The output from the previous layer.
        nb_filters: An integer representing the number of filters in X.
        compression: The compression factor for the transition layer.

    Returns:
        The output of the transition layer and the number of filters within
        the output, respectively.
    """
    pass
