#!/usr/bin/env python3
'''
'''

import tensorflow.keras as K


def one_hot(labels, classes=None):
    '''Converts a label vector into a one-hot matrix:

    Args.
        labels: A vector of labels.
        classes: Dimension for the one-hot matrix.

    Returns.
        A one-hot matrix.
    '''

    return (K.utils.to_categorical(labels, classes))
