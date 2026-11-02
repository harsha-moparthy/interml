import numpy as np


def q523(queries, documents):
    """Return batched cosine similarities [N,M] from float arrays [N,D]/[M,D]. Define a
    zero vector's similarity as zero; do not modify inputs.
    """
    q = np.asarray(queries, float)
    d = np.asarray(documents, float)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    return q @ d.T
