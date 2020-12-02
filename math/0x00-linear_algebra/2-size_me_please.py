#!/usr/bin/env python3


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


def matrix_shape(matrix):
    '''Calls get_shape to find the shape of the matrix

    Args:
        matrix: multidimensional matrix in need of shape finding
    '''
    shape = []
    get_shape(matrix, shape)
    return shape
