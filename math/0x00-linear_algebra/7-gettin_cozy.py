#!/usr/bin/env python3
'''Concatenates two matrices by a specific axis
'''


def cat_matrices2D(mat1, mat2, axis=0):
    '''Concatenates two matrices by a specific axis.

    Args:
        mat1, mat2: Matrices to concatenate.
        axis: Axis to concatenate, 0 are cols and 1 are rows

    Returns:
        A new concatenated matrix or None if the matrices can't be concatenated
    '''

    cat_matrix = []
    if axis == 1:
        if len(mat1) != len(mat2):
            return None
        for el1, el2 in zip(mat1, mat2):
            conc_mat = [x for x in el1]
            conc_mat.extend([x for x in el2])
            cat_matrix.append(conc_mat)
        return cat_matrix
    elif axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        cat_matrix.extend([[x for x in row] for row in mat1])
        cat_matrix.extend([[x for x in row] for row in mat2])
        return cat_matrix
    else:
        return None
