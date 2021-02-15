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

    init = K.initializers.he_normal()
    X = K.layers.Input(shape=(224, 224, 3))

    Y = K.layers.Conv2D(filters=64,
                        kernel_size=7,
                        strides=2,
                        padding="same",
                        kernel_initializer=init)(X)
    Y = K.layers.BatchNormalization()(Y)
    Y = K.layers.Activation("relu")(Y)
    Y = K.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(Y)
    Y = projection_block(A_prev=Y, filters=(64, 64, 256), s=1)
    Y = identity_block(A_prev=Y, filters=(64, 64, 256))
    Y = identity_block(A_prev=Y, filters=(64, 64, 256))
    Y = projection_block(A_prev=Y, filters=(128, 128, 512), s=2)
    Y = identity_block(A_prev=Y, filters=(128, 128, 512))
    Y = identity_block(A_prev=Y, filters=(128, 128, 512))
    Y = identity_block(A_prev=Y, filters=(128, 128, 512))
    Y = projection_block(A_prev=Y, filters=(256, 256, 1024), s=2)
    Y = identity_block(A_prev=Y, filters=(256, 256, 1024))
    Y = identity_block(A_prev=Y, filters=(256, 256, 1024))
    Y = identity_block(A_prev=Y, filters=(256, 256, 1024))
    Y = identity_block(A_prev=Y, filters=(256, 256, 1024))
    Y = identity_block(A_prev=Y, filters=(256, 256, 1024))
    Y = projection_block(A_prev=Y, filters=(512, 512, 2048), s=2)
    Y = identity_block(A_prev=Y, filters=(512, 512, 2048))
    Y = identity_block(A_prev=Y, filters=(512, 512, 2048))
    Y = K.layers.AveragePooling2D(pool_size=7, padding="same")(Y)
    Y = K.layers.Dense(units=1000,
                       activation="softmax",
                       kernel_initializer=init)(Y)
    model = K.models.Model(X, Y)
    return model
