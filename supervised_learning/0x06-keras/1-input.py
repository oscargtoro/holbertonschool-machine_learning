#!/usr/bin/env python3
'''Module for the function
build_model(nx, layers, activations, lambtha, keep_prob).
'''

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    '''builds a neural network with the Keras library.

    Args.
        nx: The number of input features to the network.
        layers: A list containing the number of nodes in each layer of
        the network.
        activations: A list containing the activation functions used for
        each layer of the network.
        lambtha: The L2 regularization parameter.
        keep_prob: The probability that a node will be kept for dropout.
    Returns.
        A keras model.
    '''

    l2 = K.regularizers.l2(lambtha)
    n_layers = len(layers)
    inputs = K.Input(shape=(nx,))
    x = K.layers.Dense(layers[0], activation=activations[0])(inputs)

    for i in range(n_layers):
        if i == 0:
            x = K.layers.Dropout(1 - keep_prob)(x)
        elif i < n_layers - 1:
            x = K.layers.Dense(layers[i], activation=activations[i])(x)
            x = K.layers.Dropout(1 - keep_prob)(x)
        else:
            x = K.layers.Dense(layers[i], activation=activations[i])(x)

    model = K.Model(inputs, x)
    return model
