#!/usr/bin/env python3
"""Module for the function baum_welch
"""

import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """Performs the Baum-Welch algorithm for a hidden markov model:

    Args.
        Observations: is a numpy.ndarray of shape (T,) that contains the index
        of the observation
            -T is the number of observations
        Transition: is a numpy.ndarray of shape (M, M) that contains the
        initialized transition probabilities
            -M is the number of hidden states
        Emission: is a numpy.ndarray of shape (M, N) that contains the
        initialized emission probabilities
            -N is the number of output states
        Initial: is a numpy.ndarray of shape (M, 1) that contains the
        initialized starting probabilities
        iterations: is the number of times expectation-maximization should be
        performed

    Returns.
        The converged Transition, Emission, or None, None on failure
    """
    return None, None
