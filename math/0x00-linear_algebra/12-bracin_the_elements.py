#!/usr/bin/env python3
'''Performs element-wise addition, substraction, multiplication
and division.
'''


def np_elementwise(mat1, mat2):
    '''Performs element-wise addition, substraction, multiplication
    and division of two numpy ndarrays.

    Args:
        mat1, mat2: numpy ndarrys.

    Returns:
        The element-wise addition, substraction, multiplication
        and division in that order.
    '''

    add_mat = mat1 + mat2
    sub_mat = mat1 - mat2
    mul_mat = mat1 * mat2
    div_mat = mat1 / mat2

    return add_mat, sub_mat, mul_mat, div_mat
