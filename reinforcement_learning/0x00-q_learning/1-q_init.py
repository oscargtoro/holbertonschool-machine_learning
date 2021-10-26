#!/usr/bin/env python3
"""
Module for the function(s) q_init(env)
"""

import numpy as np


def q_init(env):
    """
    Initializes the Q-table.

    Args.
        env: a FronzenLakeEnv instance.

    Returns.
        The Q-table (a numpy array of zeros) of shape (states, actions).
    """

    Q = np.zeros((env.observation_space.n, env.action_space.n))

    return Q
