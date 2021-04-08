#!/usr/bin/env python3
"""Module for the function
"""

import numpy as np


def absorbing(P):
    """Determines if a markov chain is absorbing

    Args.
        P: a square 2D numpy.ndarray of shape (n, n) representing the
        standard transition matrix
            -P[i, j] is the probability of transitioning from state i to
             state j
            -n is the number of states in the markov chain
    Returns.
        True if it is absorbing, or False on failure
    """
    return False
