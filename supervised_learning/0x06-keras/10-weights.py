#!/usr/bin/env python3
'''Module for the functions
save_weights(network, filename, save_format='h5')
load_weights(network, filename)
'''

import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    '''Saves a model’s weights.

    Args:
        network: The model whose weights should be saved.
        filename: The path of the file that the weights should be saved to.
        save_format: The format in which the weights should be saved.

    Returns:
        None
    '''

    network.save_weights(filepath=filename, save_format=save_format)
    return None


def load_weights(network, filename):
    '''Loads a model’s weights.

    Args:
        network: The model to which the weights should be loaded.
        filename: The path of the file that the weights should be loaded from.

    Returns:
        None
    '''

    network.load_weights(filepath=filename)
    return None
