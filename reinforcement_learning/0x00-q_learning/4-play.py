#!/usr/bin/env python3
"""
Module for the function(s)
    play(env, Q, max_steps=100)
"""

import numpy as np


def play(env, Q, max_steps=100):
    """
    Has the trained agent play an episode.

    Args.
        env: The FrozenLakeEnv instance.
        Q: numpy.ndarray containing the Q-table.
        max_steps: The maximum number of steps in the episode.

    Returns.
        The total rewards for the episode.
    """

    state = env.reset()

    for _ in range(max_steps):
        env.render()
        action = np.argmax(Q[state, :])
        new_state, reward, done, _ = env.step(action)
        if done is True:
            env.render()
            break
        state = new_state
    env.close()

    return reward
