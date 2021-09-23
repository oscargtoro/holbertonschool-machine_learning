#!/usr/bin/env python3
"""Module for the method positional_encoding
"""

import numpy as np


def positional_encoding(max_seq_len, dm):
    """Calculates the positional encoding for a transformer.

    Args.
        max_seq_len: Integer representing the maximum sequence length.
        dm: the model depth.

    Returns.
        A numpy.ndarray of shape (max_seq_len, dm) containing the positional
        encoding vectors.
    """

    position = np.arange(max_seq_len)
    angle_rates = 1 / np.power(
        10000, (2 * (np.arange(dm)[np.newaxis, :] // 2)) / np.float32(dm))
    PE = position[:, np.newaxis] * angle_rates
    PE[:, 0::2] = np.sin(PE[:, 0::2])
    PE[:, 1::2] = np.cos(PE[:, 1::2])

    return PE
