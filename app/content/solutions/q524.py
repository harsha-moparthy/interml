import numpy as np


def q523(queries, documents):
    q = np.asarray(queries, float)
    d = np.asarray(documents, float)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    return q @ d.T


def q524(queries, documents, k, chunk=64, exclude=None):
    """Find exact cosine top-k gallery row IDs without allocating N*M scores. Scan corpus
    chunks and keep N*k candidates. Break ties by gallery row ID. Optional exclude[N]
    omits one ID per query; k is clamped to available documents. Return (IDs,scores).
    """
    q = np.asarray(queries, float)
    d = np.asarray(documents, float)
    if k < 1 or chunk < 1 or (not len(d)):
        raise ValueError("invalid search size")
    exclusions = np.full(len(q), -1) if exclude is None else np.asarray(exclude)
    counts = len(d) - ((exclusions >= 0) & (exclusions < len(d)))
    k = min(k, int(counts.min()))
    best_scores = np.empty((len(q), 0))
    best_indices = np.empty((len(q), 0), dtype=int)
    # Merge chunk-local candidates with the running best set to bound score storage.
    for start in range(0, len(d), chunk):
        score = q523(q, d[start : start + chunk])
        ids = np.broadcast_to(np.arange(start, start + score.shape[1]), score.shape)
        score = np.where(ids == exclusions[:, None], -np.inf, score)
        score = np.concatenate((best_scores, score), 1)
        ids = np.concatenate((best_indices, ids), 1)
        order = np.lexsort((ids, -score), axis=1)[:, :k]
        best_scores = np.take_along_axis(score, order, 1)
        best_indices = np.take_along_axis(ids, order, 1)
    return (best_indices, best_scores)
