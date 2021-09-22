#!/usr/bin/env python3
"""Module for the function viterbi
"""

import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """Calculates the most likely sequence of hidden states for a hidden markov
    model

    Args.
        Observation: is a numpy.ndarray of shape (T,) that contains the index
        of the observation
            -T is the number of observations
        Emission: is a numpy.ndarray of shape (N, M) containing the emission
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

    Returns. path, P, or None, None on failure
        a list of length T containing the most likely sequence of hidden states
        and the probability of obtaining the path sequence, or None, None on
        failure
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

    viterbi = np.zeros((N, T))
    backpointer = np.zeros((N, T))

    viterbi[:, 0] = Initial.transpose() * Emission[:, Observation[0]]

    for t in range(1, T):
        a = viterbi[:, t - 1]
        b = Transition.transpose()
        ab = a * b
        ab_max = np.amax(ab, axis=1)
        c = Emission[:, Observation[t]]
        prob = ab_max * c

        viterbi[:, t] = prob
        backpointer[:, t - 1] = np.argmax(ab, axis=1)

    path = []
    current = np.argmax(viterbi[:, T - 1])
    path = [current] + path

    for t in range(T - 2, -1, -1):
        current = int(backpointer[current, t])
        path = [current] + path

    P = np.amax(viterbi[:, T - 1], axis=0)

    return path, P
