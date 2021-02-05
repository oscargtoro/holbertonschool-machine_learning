#!/usr/bin/env python3
'''Module for the function
lenet5(x, y).
'''

import tensorflow as tf


def lenet5(x, y):
    '''Modified version of the LeNet-5 architecture using tensorflow.

    Args:
        x: tf.placeholder of shape (m, 28, 28, 1) containing the input images
        for the network
            -m: The number of images.
        y: tf.placeholder of shape (m, 10) containing the one-hot labels for
        the network.

    Returns:
    A tensor for the softmax activated output, a training operation that
    utilizes Adam optimization (with default hyperparameters), a tensor
    for the loss of the netowrk, a tensor for the accuracy of the network.
    '''
    pass
