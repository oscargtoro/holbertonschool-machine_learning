#!/usr/bin/env python3
"""Module for the function create_momentum_op
"""

import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """Creates the training operation for a neural network in tensorflow using
    the gradient descent with momentum optimization algorithm

    Args:
        loss: is the loss of the network
        alpha: is the learning rate
        beta1: is the momentum weight

    Returns:
        The momentum optimization operation
    """
    return tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)
