#!/usr/bin/env python3
'''Returns the transpose of a 2D matrix.
'''


def get_shape(matrix, shape):
    '''Recoursively finds the shape of a multidimensional matrix.

    Args:
        matrix: multidimensional matrix in need of shape finding.
        shape: empty list to store shape.
    '''
    if type(matrix[0]) is list:
        shape.append(len(matrix))
        get_shape(matrix[0], shape)
    else:
        shape.append(len(matrix))


def matrix_transpose(matrix):
    '''Transpose a given matrix

    Args:
        matrix: matrix to transpose.

    Returns:
        A list representing the transpose matrix.
    '''
    tmatrix = []
    shape = []
    get_shape(matrix, shape)
    for col in range(shape[1]):
        # tmatrix.append(list[x for x in matrix[]])
        tmatrix.append(list([matrix[row][col] for row in range(shape[0])]))
    return tmatrix
