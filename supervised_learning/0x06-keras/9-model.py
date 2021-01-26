#!/usr/bin/env python3
'''Module for the functions
save_model(network, filename)
load_model(filename)
'''

import tensorflow.keras as K


def save_model(network, filename):
    '''Saves an entire model.

    Args:
        network: The model to save.
        filename: The path of the file that the model should be saved to.
    Returns:
        None
    '''

    network.save(filepath=filename)
    return None


def load_model(filename):
    '''Loads an entire model.
    Args:
        filename: The path of the file that the model should be loaded from.

    Returns:
        The loaded model
    '''

    return K.models.load_model(filepath=filename)
