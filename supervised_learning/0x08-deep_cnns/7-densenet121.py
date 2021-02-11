#!/usr/bin/env python3
"""Module for the function densenet121 that builds the DenseNet-121
architecture as described in Densely Connected Convolutional Networks.
https://arxiv.org/pdf/1608.06993.pdf
"""

import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """Builds the DenseNet-121 architecture as described in Densely
    Connected Convolutional Networks.

    Args:
        growth_rate: Is the growth rate.
        compression: Is the compression factor.

    Returns:
        The keras model.
    """
    pass
