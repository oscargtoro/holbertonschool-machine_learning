#!/usr/bin/env python3
"""Module for the function inception_network that builds the inception network
as described in Going Deeper with Convolutions (2014).
"""

import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """Builds the inception network as described in Going Deeper with
    Convolutions (2014).

    Returns:
        The keras model of the inception network.
    """
    pass
