#!/usr/bin/env python3
"""Module for the function resnet50 that builds the ResNet-50 architecture as
described in Deep Residual Learning for Image Recognition (2015)
https://arxiv.org/pdf/1512.03385.pdf
"""

import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Builds the ResNet-50 architecture as described in Deep Residual
    Learning for Image Recognition (2015).

    Returns:
        The keras model
    """
    pass
