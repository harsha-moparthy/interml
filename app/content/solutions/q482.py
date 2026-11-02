import numpy as np


def q482(query_embeddings, candidate_embeddings, k=5):
    """Rank candidates by cosine similarity to each query. Normalize last-axis vectors with
    epsilon protection and return top-k indices and scores, descending with stable ties.
    """
    q = np.asarray(query_embeddings, float)
    c = np.asarray(candidate_embeddings, float)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    c = c / np.maximum(np.linalg.norm(c, axis=-1, keepdims=True), 1e-12)
    score = q @ c.T
    index = np.argsort(-score, axis=-1, kind="stable")[:, :k]
    return (index, np.take_along_axis(score, index, axis=-1))
