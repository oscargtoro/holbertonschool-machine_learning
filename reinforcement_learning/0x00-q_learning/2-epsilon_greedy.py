#!/usr/bin/env python3
"""
Module for the function(s) epsilon_greedy(Q, state, epsilon)
"""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Uses epsilon-greedy to determine the next action.

    Args.
        Q: numpy.ndarray containing the q-table.
        state: the current state.
        epsilon: treshold for exploration or exploitation

    Returns.
        The next action index
    """

    if np.random.uniform(0, 1) < epsilon:
        a = np.random.randint(0, Q.shape[1])
    else:
        a = np.argmax(Q[state, :])

    return a
