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
    if not isinstance(P, np.ndarray):
        raise TypeError("P must be a 1D numpy.ndarray")
    if not all([0 <= x <= 1 for x in P]):
        raise ValueError("All values in P must be in the range [0, 1]")
    pass


def intersection(x, n, P, Pr):
    """Calculates the intersection of obtaining this data with the various
    hypothetical probabilities

    Args.
        x: is the number of patients that develop severe side effects
        n: is the total number of patients observed
        P: is a 1D numpy.ndarray containing the various hypothetical
        probabilities of developing severe side effects
        Pr: is a 1D numpy.ndarray containing the prior beliefs of P

    Returns.
        A 1D numpy.ndarray containing the intersection of obtaining x and n
        with each probability in P, respectively
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
    if not isinstance(Pr, np.ndarray) or len(Pr.shape) != len(P.shape):
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    P_valtest = all([0 <= x <= 1 for x in P])
    Pr_valtest = all([0 <= x <= 1 for x in Pr])
    if not P_valtest:
        msg = "All values in {} must be in the range [0, 1]".format("P")
        raise ValueError(msg)
    if not Pr_valtest:
        msg = "All values in {} must be in the range [0, 1]".format("Pr")
        raise ValueError(msg)
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")
    pass


def marginal(x, n, P, Pr):
    """Calculates the marginal probability of obtaining the data

    Args.
        x: is the number of patients that develop severe side effects
        n: is the total number of patients observed
        P: is a 1D numpy.ndarray containing the various hypothetical
        probabilities of patients developing severe side effects
        Pr: is a 1D numpy.ndarray containing the prior beliefs about P

    Returns.
        The marginal probability of obtaining x and n
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
    if not isinstance(Pr, np.ndarray) or len(Pr.shape) != len(P.shape):
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    P_valtest = all([0 <= x <= 1 for x in P])
    Pr_valtest = all([0 <= x <= 1 for x in Pr])
    if not P_valtest:
        msg = "All values in {} must be in the range [0, 1]".format("P")
        raise ValueError(msg)
    if not Pr_valtest:
        msg = "All values in {} must be in the range [0, 1]".format("Pr")
        raise ValueError(msg)
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")
    pass


def posterior(x, n, P, Pr):
    """calculates the posterior probability for the various hypothetical
    probabilities of developing severe side effects given the data:

    Args.
        x: is the number of patients that develop severe side effects
        n: is the total number of patients observed
        P: is a 1D numpy.ndarray containing the various hypothetical
        probabilities of patients developing severe side effects
        Pr: is a 1D numpy.ndarray containing the prior beliefs about P

    Returns.
        The marginal probability of obtaining x and n
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
    if not isinstance(Pr, np.ndarray) or len(Pr.shape) != len(P.shape):
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    P_valtest = all([0 <= x <= 1 for x in P])
    Pr_valtest = all([0 <= x <= 1 for x in Pr])
    if not P_valtest:
        msg = "All values in {} must be in the range [0, 1]".format("P")
        raise ValueError(msg)
    if not Pr_valtest:
        msg = "All values in {} must be in the range [0, 1]".format("Pr")
        raise ValueError(msg)
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")
    pass
