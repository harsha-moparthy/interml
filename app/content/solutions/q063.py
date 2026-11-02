"""Compute the change-of-basis matrix between the supplied bases B and C."""

import numpy as np


def transform_basis(B, C):
    C = np.array(C)
    B = np.array(B)
    C_inv = np.linalg.inv(C)
    P = np.dot(C_inv, B)
    return P.tolist()
