#!/usr/bin/env python3
"""Module for the function f1_score
Calculates the F1 score of a confusion matrix
"""

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """Calculates the F1 score of a confusion matrix

    Args.
        confusion: is a confusion numpy.ndarray of shape (classes, classes)
        where row indices represent the correct labels and column indices
        represent the predicted labels
    Returns:
    A numpy.ndarray of shape (classes,) containing the F1 score of each class
    """

    PPV = precision(confusion)
    TPR = sensitivity(confusion)
    return 2 * (PPV * TPR) / (PPV + TPR)
