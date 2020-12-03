#!/usr/bin/env python3
'''Adds two matrices element-wise
'''


def get_shape(matrix, shape):
    '''Recoursively finds the shape of a multidimensional matrix

    Args:
        matrix: multidimensional matrix in need of shape finding
        shape: empty list to store shape
    '''
    if type(matrix[0]) is list:
        shape.append(len(matrix))
        get_shape(matrix[0], shape)
    else:
        shape.append(len(matrix))


def add_matrices2D(mat1, mat2):
    '''Adds two matrices element-wise

    Args:
        mat1, mat2: Matrices to add element-wise

    Returns:
        added_mat: Matrix with with the element-wise addition result
    '''

    added_mat = []
    mat1_shape = []
    mat2_shape = []
    get_shape(mat1, mat1_shape)
    get_shape(mat2, mat2_shape)

    # Compare matrices shapes if different returns None
    for el1, el2 in zip(mat1_shape, mat2_shape):
        if el1 != el2:
            return None

    # Adds the two matrices element-wise
    for row_idx in range(mat1_shape[0]):
        arr = []
        for col_idx in range(mat1_shape[1]):
            arr.append(mat1[row_idx][col_idx] + mat2[row_idx][col_idx])
        added_mat.append(arr)
    return added_mat
