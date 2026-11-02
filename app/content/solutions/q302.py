import numpy as np


def q302(word, subword_vectors, min_n=3, max_n=6):
    """Compute a FastText-style word representation by averaging available character n-gram
    vectors from boundary-marked text. Return zero if no known n-gram exists. The vector
    dictionary must be nonempty.
    """
    wrapped = "<" + word + ">"
    vectors = []
    for n in range(min_n, max_n + 1):
        for i in range(len(wrapped) - n + 1):
            token = wrapped[i : i + n]
            if token in subword_vectors:
                vectors.append(subword_vectors[token])
    return (
        np.mean(vectors, axis=0)
        if vectors
        else np.zeros_like(next(iter(subword_vectors.values())), dtype=float)
    )
