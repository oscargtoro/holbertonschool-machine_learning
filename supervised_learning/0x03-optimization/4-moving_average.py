#!/usr/bin/env python3
"""Module for the function moving_average
"""

import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def moving_average(data, beta):
    """Calculates the weighted moving average of a data set

    Args:
        data: is the list of data to calculate the moving average of
        beta: is the weight used for the moving average
    Returns:
        a list containing the moving averages of data
    """

    mov_avg = []
    val = 0
    for i in range(len(data)):
        val = beta * val + (1 - beta) * data[i]
        mov_avg.append(val / (1 - (beta ** (i + 1))))
    return mov_avg
