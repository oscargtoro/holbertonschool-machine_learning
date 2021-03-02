#!/usr/bin/env python3
"""Module for the function update_variables_RMSProp
"""

import tensorflow as tf


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """Updates a variable using the RMSProp optimization algorithm:

    Args:
        alpha: is the learning rate
        beta2: is the RMSProp weight
        epsilon: is a small number to avoid division by zero
        var: is a numpy.ndarray containing the variable to be updated
        grad: is a numpy.ndarray containing the gradient of var
        s: is the previous second moment of var

    Returns:
        The updated variable and the new moment, respectively
    """

    nm = beta2 * s + (1 - beta2) * (grad ** 2)
    up_var = var - alpha * grad / ((nm ** 0.5) + epsilon)
    return up_var, nm
