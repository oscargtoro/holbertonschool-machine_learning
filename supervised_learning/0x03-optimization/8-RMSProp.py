#!/usr/bin/env python3
"""Module for the function create_RMSProp_op
"""

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """Creates the training operation for a neural network in tensorflow using
    the RMSProp optimization algorithm:

    Args:
        loss: is the loss of the network
        alpha: is the learning rate
        beta2: is the RMSProp weight
        epsilon: is a small number to avoid division by zero
    Returns:
        The RMSProp optimization operation
    """

    return tf.train.RMSPropOptimizer(alpha, beta2, 0, epsilon).minimize(loss)
