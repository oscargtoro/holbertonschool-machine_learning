#!/usr/bin/env python3
"""Module for the function
"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """Performs the forward algorithm for a hidden markov model

    Args.
        Observation: a numpy.ndarray of shape (T,) that contains the index of
        the observation
            -T is the number of observations
        Emission: a numpy.ndarray of shape (N, M) containing the emission
        probability of a specific observation given a hidden state
            -Emission[i, j] is the probability of observing j given the hidden
             state i
            -N is the number of hidden states
            -M is the number of all possible observations
        Transition: is a 2D numpy.ndarray of shape (N, N) containing the
        transition probabilities
            -Transition[i, j] is the probability of transitioning from the
             hidden state i to j
        Initial: a numpy.ndarray of shape (N, 1) containing the probability of
        starting in a particular hidden state

    Returns:
        The likelihood of the observations given the model and a numpy.ndarray
        of shape (N, T) containing the forward path probabilities, or None,
        None on failure
    """

    if not isinstance(Observation, np.ndarray) or len(Observation.shape) != 1:
        return None, None

    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None

    if not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2:
        return None, None

    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
        return None, None

    T = Observation.shape[0]

    N, _ = Emission.shape

    if Transition.shape[0] != N or Transition.shape[1] != N:
        return None, None

    if Initial.shape[0] != N or Initial.shape[1] != 1:
        return None, None

    if not np.sum(Emission, axis=1).all():
        return None, None

    if not np.sum(Transition, axis=1).all():
        return None, None

    if not np.sum(Initial) == 1:
        return None, None

    F = np.zeros((N, T))

    F[:, 0] = Initial.transpose() * Emission[:, Observation[0]]

    for i in range(1, T):
        F[:,
          i] = np.matmul(F[:, i - 1], Transition) * Emission[:, Observation[i]]

    P = np.sum(F[:, T - 1])

    return P, F
