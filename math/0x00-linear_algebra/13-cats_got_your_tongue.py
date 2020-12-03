#!/usr/bin/env python3
'''Concatenates two numpy ndarrays along a specific axis.
'''

import numpy as np


def np_cat(mat1, mat2, axis=0):
    '''Concatenates two numpy ndarrays along a specific axis.

    Args:
        mat1, mat2:
            numpy ndarrays.
        axis:
            integer that represents the axis.

    Returns:
        matrix with mat1 and mat2 concatenated.
    '''

    return np.concatenate((mat1, mat2), axis)
