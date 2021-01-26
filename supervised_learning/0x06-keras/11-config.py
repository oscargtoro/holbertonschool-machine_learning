#!/usr/bin/env python3
'''Module for functions
save_config(network, filename)
load_config(filename)
'''

import tensorflow.keras as K


def save_config(network, filename):
    '''Saves a model’s configuration in JSON format.

    Args:
        network: The model whose configuration should be saved.
        filename: The path of the file that the configuration should
        be saved to

    Returns:
        None
    '''

    with open(filename, 'w') as f:
        f.write(network.to_json())

    return None


def load_config(filename):
    '''Loads a model with a specific configuration.

    Args:
        filename: The path of the file containing the model’s configuration
        in JSON format.

    Returns:
        The loaded model
    '''

    with open(filename) as f:
        model = K.models.model_from_json(f.read())

    return model
