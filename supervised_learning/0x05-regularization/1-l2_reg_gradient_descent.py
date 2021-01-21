#!/usr/bin/env python3
'''Module for the function
l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L)
'''

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    '''Updates the weights and biases of a neural network using
    gradient descent with L2 regularization (The neural network
    uses tanh activations on each layer except the last, which
    uses a softmax activation, the weights and biases of the
    network are updated in place).

    Args.
        Y: A one-hot numpy.ndarray of shape (classes, m) that contains the
        correct labels for the data.
            - classes: The number of classes.
            - m: The number of data points.
        weights: A dictionary of the weights and biases of the NN.
        cache: A dictionary of the outputs of each layer of the NN.
        alpha: The learning rate.
        lambtha: The L2 regularization parameter.
        L: The number of layers of the network.
    '''

    m = Y.shape[1]
    dz = cache["A3"] - Y
    for i in range(L - 1, 0, -1):
        db = 0
        dw = 0
        A_i = cache["A{}".format(i - 1)]
        W_i = weights["W{}".format(i)]
        b_i = weights["b{}".format(i)]
        dza = np.matmul(np.transpose(weights['W{}'.format(i + 1)]), dz)
        dzb = 1 - cache['A{}'.format(i)] ** 2
        dz = dza * dzb

        dw = ((1 / m) * np.matmul(dz, np.transpose(A_i))) + \
            (lambtha / m) * W_i
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True) + (lambtha / m) * b_i
        weights["W{}".format(i + 1)] = W_i - alpha * dw
        weights["b{}".format(i + 1)] = b_i - alpha * db
