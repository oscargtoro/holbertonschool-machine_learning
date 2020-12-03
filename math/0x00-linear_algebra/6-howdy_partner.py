#!/usr/bin/env python3
'''Concatenates two arrays.
'''


def cat_arrays(arr1, arr2):
    '''Concatenates two arrays.

    Args:
        arr1, arr2: Arrays to concatenate.

    Returns:
        New array with arr1 and arr2 elements.
    '''

    ext_array = arr1.copy()
    ext_array.extend(arr2)
    return ext_array
