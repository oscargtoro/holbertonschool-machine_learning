#!/usr/bin/env python3
"""Module for the function likelihood
    - likelihood: Calculates the likelihood of obtaining this data given
    various hypothetical probabilities of developing severe side effects
"""

import numpy as np


def likelihood(x, n, P):
    """Calculates the likelihood of obtaining this data given various
    hypothetical probabilities of developing severe side effects

    Args.
        x: is the number of patients that develop severe side effects
        n: is the total number of patients observed
        P: is a 1D numpy.ndarray containing the various hypothetical
        probabilities of developing severe side effects

    Returns.
        A 1D numpy.ndarray containing the likelihood of obtaining the data,
        x and n, for each probability in P, respectively
    """

    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        msg = "x must be an integer that is greater than or equal to 0"
        raise ValueError(msg)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not all([0 <= x <= 1 for x in P]):
        raise ValueError("All values in P must be in the range [0, 1]")
    pass
