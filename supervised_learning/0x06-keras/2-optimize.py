#!/usr/bin/env python3
'''Module for the function optimize_model(network, alpha, beta1, beta2)
'''

import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    '''sets up Adam optimization for a keras model with categorical
    crossentropy loss and accuracy metrics.

    Args.
        network: The model to optimize.
        alpha: The learning rate.
        beta1: The first Adam optimization parameter.
        beta2: The second Adam optimization parameter

    Returns.
        None
    '''

    adam = K.optimizers.Adam(lr=alpha, beta_1=beta1, beta_2=beta2)
    network.compile(loss='categorical_crossentropy',
                    optimizer=adam,
                    metrics=['accuracy'])
    return None
