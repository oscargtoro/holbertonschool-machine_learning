#!/usr/bin/env python3
'''Module for the function
test_model(network, data, labels, verbose=True)
'''

import tensorflow.keras as k


def test_model(network, data, labels, verbose=True):
    '''Tests a neural network.

    Args:
        network: The network model to test.
        data: The input data to test the model with.
        labels: The correct one-hot labels of data.
        verbose: boolean that determines if output should be printed during
        the testing process.

    Returns:
    The loss and accuracy of the model with the testing data, respectively.
    '''

    results = network.evaluate(x=data, y=labels, verbose=verbose)
    return results
