#!/usr/bin/env python3
"""Module for the function create_batch_norm_layer
"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer for a neural network in tensorflow

    Args.
        prev: is the activated output of the previous layer
        n: is the number of nodes in the layer to be created
        activation: is the activation function that should be used on the
        output of the layer

    Returns:
        A tensor of the activated output for the layer
    """

    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    base_layer = tf.layers.Dense(units=n, kernel_initializer=init)
    X = base_layer(prev)
    u, var = tf.nn.moments(X, axes=[0])
    gamma = tf.Variable(tf.constant(1.0, shape=(1, n)), trainable=True)
    beta = tf.Variable(tf.constant(0.0, shape=(1, n)), trainable=True)

    Z = tf.nn.batch_normalization(x=X, mean=u, variance=var,
                                  offset=beta, scale=gamma,
                                  variance_epsilon=1e-8,
                                  name='Z')

    return activation(Z)
