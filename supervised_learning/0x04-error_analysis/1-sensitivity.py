#!/usr/bin/env python3
"""Module for the function sensitivity
"""

import numpy as np


def sensitivity(confusion):
    """Calculates the sensitivity for each class in a confusion matrix

    Args.
        confusion: A confusion numpy.ndarray of shape (classes, classes) where
        row indices represent the correct labels and column indices represent
        the predicted labels
        classes: is the number of classes
    Returns:
        A numpy.ndarray of shape (classes,) containing the sensitivity
        of each class
    """

    sen = np.zeros((confusion.shape[0]))

    for i in range(confusion.shape[0]):
        sen[i] = (confusion[i][i] / np.sum(confusion[i, :]))
    return sen
