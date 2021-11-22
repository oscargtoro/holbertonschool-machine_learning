#!/usr/bin/env python3
'''
Module for the function(s):
    - from_numpy(array): creates a pd.DataFrame from an np.ndarray
'''

import pandas as pd


def from_numpy(array):
    '''
    Creates a pd.DataFrame from an np.ndarray.

    Args.
        array: The np.ndarray from wich the pd.DataFrame will be created.

    Returns.
        A newly created pd.DataFrame.
    '''

    # Creates an alphabet list to use as columns using the shape of the array
    columns = list(map(chr, range(65, 65 + array.shape[1])))
    return pd.DataFrame(array, columns=columns)
