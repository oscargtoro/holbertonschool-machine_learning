#!/usr/bin/env python3
'''
Module for the function(s):
    - from_file(filename, delimiter): loads data from a file as a pd.DataFrame
'''

import pandas as pd


def from_file(filename, delimiter):
    '''
    Loads data from a file as a pd.DataFrame.

    Args.
        filename: The file to load from.
        delimiter: The column separator.

    Returns.
    The loaded pd.DataFrame
    '''

    return pd.read_csv(filename, delimiter=delimiter)
