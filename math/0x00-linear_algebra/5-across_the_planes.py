#!/usr/bin/env python3
'''Adds two matrices element-wise
'''


def add_matrices2D(mat1, mat2):
    '''Adds two matrices element-wise

    Args:
        mat1, mat2: Matrices to add element-wise

    Returns:
        added_mat: Matrix with with the element-wise addition result
    '''

    added_mat = []

    # Compare matrices shapes if different returns None
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None

    # Adds the two matrices element-wise
    for row_idx in range(len(mat1)):
        arr = []
        for col_idx in range(len(mat1[0])):
            arr.append(mat1[row_idx][col_idx] + mat2[row_idx][col_idx])
        added_mat.append(arr)
    return added_mat
