#!/usr/bin/env python3
'''Multiplies two matrices.
'''


def mat_mul(mat1, mat2):
    '''Multiplies two matrices.

    Args:
        mat1, mat2: matrices to multiply

    Returns:
        A new matrix representing the multiplication of mat1 and mat2,
        if by matrix mult rules the two matrices can't be multiplied
        returns None
    '''

    if len(mat1[0]) != len(mat2):
        return None
    mul_mat = []
    for row in range(len(mat1)):
        mul_row = []
        for mat2_col in range(len(mat2[0])):
            mul_sum = 0
            for col in range(len(mat1[0])):
                mul_sum += (mat1[row][col] * mat2[col][mat2_col])
            mul_row.append(mul_sum)
        mul_mat.append(mul_row)
    return mul_mat
