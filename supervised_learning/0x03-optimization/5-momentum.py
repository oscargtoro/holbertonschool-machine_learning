#!/usr/bin/env python3
"""Module for the function update_variables_momentum
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Updates a variable using the gradient descent with momentum
    optimization algorithm

    Args:
        alpha: is the learning rate
        beta1: is the momentum weight
        var: is a numpy.ndarray containing the variable to be updated
        grad: is a numpy.ndarray containing the gradient of var
        v: is the previous first moment of var
    Returns:
        The updated variable and the new moment, respectively
    """
    vdW = (beta1 * v) + (1 - beta1) * grad
    W = var - (alpha * vdW)

    return W, vdW
