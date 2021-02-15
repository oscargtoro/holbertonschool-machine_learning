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

    init = K.initializers.he_normal()
    X = K.layers.Input(shape=(224, 224, 3))

    Y = K.layers.Conv2D(filters=64,
                        kernel_size=7,
                        strides=2,
                        padding="same",
                        activation="relu",
                        kernel_initializer=init)(X)
    Y = K.layers.MaxPool2D(pool_size=64,
                           strides=2,
                           padding="same")(Y)
    Y = K.layers.Conv2D(filters=64,
                        kernel_size=1,
                        strides=1,
                        padding="same",
                        activation="relu",
                        kernel_initializer=init)(Y)
    Y = K.layers.Conv2D(filters=192,
                        kernel_size=3,
                        strides=1,
                        padding="same",
                        activation="relu",
                        kernel_initializer=init)(Y)
    Y = K.layers.MaxPool2D(pool_size=192, strides=2, padding="same")(Y)
    Y = inception_block(A_prev=Y, filters=(64, 96, 128, 16, 32, 32))
    Y = inception_block(A_prev=Y, filters=(128, 128, 192, 32, 96, 64))
    Y = K.layers.MaxPool2D(pool_size=480, strides=2, padding="same")(Y)
    Y = inception_block(A_prev=Y, filters=(192, 96, 208, 16, 48, 64))
    Y = inception_block(A_prev=Y, filters=(160, 112, 224, 24, 64, 64))
    Y = inception_block(A_prev=Y, filters=(128, 128, 256, 24, 64, 64))
    Y = inception_block(A_prev=Y, filters=(112, 144, 288, 32, 64, 64))
    Y = inception_block(A_prev=Y, filters=(256, 160, 320, 32, 128, 128))
    Y = K.layers.MaxPool2D(pool_size=832, strides=2, padding="same")(Y)
    Y = inception_block(A_prev=Y, filters=(256, 160, 320, 32, 128, 128))
    Y = inception_block(A_prev=Y, filters=(384, 192, 384, 48, 128, 128))
    Y = K.layers.AveragePooling2D(pool_size=7, padding="same")(Y)
    Y = K.layers.Dropout(rate=0.4)(Y)
    Y = K.layers.Dense(units=1000,
                       activation="relu",
                       kernel_initializer=init)(Y)
    return K.models.Model(inputs=X, outputs=Y)
