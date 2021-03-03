#!/usr/bin/env python3
"""Module for the function create_confusion_matrix
"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """Creates a confusion matrix

    Args.
        labels: is a one-hot numpy.ndarray of shape (m, classes) containing
        the correct labels for each data point
        classes: is the number of classes
        logits: is a one-hot numpy.ndarray of shape (m, classes) containing
        the predicted labels
    Returns:
        A confusion numpy.ndarray of shape (classes, classes) with row indices
        representing the correct labels and column indices representing
        the predicted labels
    """

    pred = 0
    lab = 0
    cm_shape = (labels.shape[1], labels.shape[1])
    c_matrix = np.zeros(cm_shape)

    for i in range(labels.shape[0]):
        pred = ((np.where(logits[i] == 1))[0][0])
        lab = ((np.where(labels[i] == 1))[0][0])
        c_matrix[lab][pred] += 1

    return c_matrix
