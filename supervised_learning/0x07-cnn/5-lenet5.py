#!/usr/bin/env python3
'''Module for the function lenet5.
Builds a modified version of the LeNet-5 architecture using keras.
'''

import tensorflow.keras as K


def lenet5(X):
    '''Builds a modified version of the LeNet-5 architecture using keras.

    Args:
        x: keras.Input of shape (m, 28, 28, 1) containing the input images
        for the network
            -m: The number of images.

    Returns:
        A keras Model compiled to use Adam optimization (with default
        hyperparameters) and accuracy metrics
    '''
    init = K.initializers.he_normal(seed=None)

    x = K.layers.Conv2D(filters=6,
                        kernel_size=5,
                        padding='same',
                        activation='relu',
                        kernel_initializer=init)(X)

    x = K.layers.MaxPool2D(pool_size=2, strides=(2, 2))(x)

    x = K.layers.Conv2D(filters=16,
                        kernel_size=5,
                        padding='valid',
                        activation='relu',
                        kernel_initializer=init)(x)

    x = K.layers.MaxPool2D(pool_size=2, strides=(2, 2))(x)

    x = K.layers.Flatten()(x)

    x = K.layers.Dense(units=120,
                       activation='relu',
                       kernel_initializer=init)(x)

    x = K.layers.Dense(units=84,
                       activation='relu',
                       kernel_initializer=init)(x)

    output = K.layers.Dense(units=10,
                            activation='softmax',
                            kernel_initializer=init)(x)

    model = K.Model(inputs=X, outputs=output)

    Adam = K.optimizers.Adam()

    model.compile(optimizer=Adam,
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    return model
