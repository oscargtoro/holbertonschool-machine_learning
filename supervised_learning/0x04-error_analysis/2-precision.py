#!/usr/bin/env python3
"""Module for the function precision
Calculates the precision for each class in a confusion matrix
"""

import numpy as np


def precision(confusion):
    """Calculates the precision for each class in a confusion matrix

    Args.
        confusion: is a confusion numpy.ndarray of shape (classes, classes)
        where row indices represent the correct labels and column indices
        represent the predicted labels
    Returns:
        A numpy.ndarray of shape (classes,) containing the precision
        of each class
    """

    prec = np.zeros((confusion.shape[0],))

    for i in range(confusion.shape[1]):
        prec[i] = confusion[i][i] / np.sum(confusion[:, i])
    return prec
