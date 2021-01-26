#!/usr/bin/env python3
'''Module for the function
predict(network, data, verbose=False)
'''

import tensorflow.keras as K


def predict(network, data, verbose=False):
    '''Makes a prediction using a neural network.

    Args:
        network: The network model to make the prediction with.
        data: The input data to make the prediction with.
        verbose: boolean that determines if output should be printed during
        the prediction process.

    Returns:
    The prediction for the data
    '''
    return network.predict(x=data, verbose=verbose)
