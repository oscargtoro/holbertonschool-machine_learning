#!/usr/bin/env python3
'''Adds two arrays element-wise.
'''


def add_arrays(arr1, arr2):
    '''Adds two arrays element-wise.

    Args:
        arr1, arr2: arrays to add.

    Returns:
        A list with the two arrays added.
    '''

    if len(arr1) > len(arr2):
        return None
    else:
        add_arr = []
        for idx in range(len(arr1)):
            add_arr.append(arr1[idx] + arr2[idx])
        return add_arr

