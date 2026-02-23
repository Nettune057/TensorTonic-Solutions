import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """

    transpose = [list(row) for row in zip(*A)]

    # Write code here
    return np.asarray(transpose)
