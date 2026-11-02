"""Solve a symmetric positive-definite linear system using conjugate gradients."""

import numpy as np


def conjugate_gradient(
    A: np.array, b: np.array, n: int, x0: np.array = None, tol=1e-08
) -> np.array:
    x = np.zeros_like(b)
    r = residual(A, b, x)
    rPlus1 = r
    p = r
    for i in range(n):
        alp = alpha(A, r, p)
        x = x + alp * p
        rPlus1 = r - alp * (A @ p)
        bet = beta(r, rPlus1)
        p = rPlus1 + bet * p
        r = rPlus1
        if np.linalg.norm(residual(A, b, x)) < tol:
            break
    return x


def residual(A: np.array, b: np.array, x: np.array) -> np.array:
    return b - A @ x


def alpha(A: np.array, r: np.array, p: np.array) -> float:
    alpha_num = np.dot(r, r)
    alpha_den = np.dot(p @ A, p)
    return alpha_num / alpha_den


def beta(r: np.array, r_plus1: np.array) -> float:
    beta_num = np.dot(r_plus1, r_plus1)
    beta_den = np.dot(r, r)
    return beta_num / beta_den
