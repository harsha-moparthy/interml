"""Reduce a matrix to reduced row echelon form using row operations."""

import numpy as np


def rref(matrix):
    A = matrix.astype(np.float32)
    n, m = A.shape
    row = 0
    for col in range(m):
        if row >= n:
            break
        nonzero_rel_id = np.nonzero(A[row:, col])[0]
        if len(nonzero_rel_id) == 0:
            continue
        k = nonzero_rel_id[0] + row
        A[[row, k]] = A[[k, row]]
        A[row] = A[row] / A[row, col]
        for j in range(n):
            if j != row:
                A[j] -= A[j, col] * A[row]
        row += 1
    return A
