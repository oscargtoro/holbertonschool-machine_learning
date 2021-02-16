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

    init = K.initializers.he_normal()
    nb_filters = 0
    X = K.layers.Input((224, 224, 3))

    Y = K.layers.BatchNormalization()(X)
    Y = K.layers.Activation(activation="relu")(Y)
    Y = K.layers.Conv2D(filters=64,
                        kernel_size=7,
                        strides=2,
                        padding="same",
                        kernel_initializer=init)(Y)
    Y = K.layers.MaxPool2D(pool_size=2, padding="same", strides=2)(Y)
    Y, nb_filters = dense_block(Y, Y.shape[3], growth_rate, 6)
    Y, nb_filters = transition_layer(Y, int(nb_filters), compression)
    Y, nb_filters = dense_block(Y, int(nb_filters), growth_rate, 12)
    Y, nb_filters = transition_layer(Y, int(nb_filters), compression)
    Y, nb_filters = dense_block(Y, Y.shape[3], growth_rate, 24)
    Y, nb_filters = transition_layer(Y, int(nb_filters), compression)
    Y, nb_filters = dense_block(Y, Y.shape[3], growth_rate, 16)
    Y = K.layers.AveragePooling2D(pool_size=7, padding="same")(Y)
    Y = K.layers.Dense(units=1000,
                       activation="softmax",
                       kernel_initializer=init)(Y)
    return K.models.Model(X, Y)
