#!/usr/bin/env python3
'''Performs matrix multiplication on two numpy ndarrays.
'''

import numpy as np


def np_matmul(mat1, mat2):
    '''Performs matrix multiplication on two numpy ndarrays.

    Args:
        mat1, mat2:
            numpy ndarrays

    Returns:
        numpy ndarray with result
    '''

    return np.matmul(mat1, mat2)
