#!/usr/bin/env python3
"""Module for the function specificity
Calculates the specificity for each class in a confusion matrix
"""

import numpy as np


def specificity(confusion):
    """Calculates the specificity for each class in a confusion matrix

    Args.
        confusion: A confusion numpy.ndarray of shape (classes, classes) where
        row indices represent the correct labels and column indices represent
        the predicted labels
    Returns: a numpy.ndarray of shape (classes,) containing the specificity
    of each class
    """

    TP = np.zeros((confusion.shape[0],))
    FP = np.zeros((confusion.shape[0],))
    TN = np.zeros((confusion.shape[0],))
    FN = np.zeros((confusion.shape[0],))

    spec = np.zeros((confusion.shape[0],))

    for i in range(confusion.shape[0]):
        TP[i] = confusion[i][i]
        cnf_masked = np.ma.array(confusion, mask=False)
        cnf_masked.mask[i] = True
        FP[i] = np.sum(cnf_masked[:, i])
        cnf_masked.mask[i] = False
        cnf_masked.mask[:, i] = True
        FN[i] = np.sum(cnf_masked[i, :])

    TN = np.sum(confusion) - TP - FP - FN
    return TN / (TN + FP)
