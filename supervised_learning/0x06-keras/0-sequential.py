#!/usr/bin/env python3
'''Module for the function
build_model(nx, layers, activations, lambtha, keep_prob)
'''

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    '''Builds a neural network with the Keras library.

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

    model = K.Sequential()
    l2 = K.regularizers.l2(lambtha)

    for i in range(len(layers)):
        if i == 0:
            model.add(K.layers.Dense(units=layers[i],
                                     activation=activations[i],
                                     kernel_regularizer=l2,
                                     input_shape=(nx,)))
        else:
            model.add(K.layers.Dense(units=layers[i],
                                     activation=activations[i],
                                     kernel_regularizer=l2))
        if i < len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
