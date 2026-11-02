import numpy as np
from collections import Counter


def q296(documents, vocabulary):
    """Construct a bag-of-words count matrix from tokenized documents and a dense
    zero-based vocabulary mapping. Ignore out-of-vocabulary tokens.
    """
    result = np.zeros((len(documents), len(vocabulary)), dtype=int)
    for i, tokens in enumerate(documents):
        for token, count in Counter(tokens).items():
            if token in vocabulary:
                result[i, vocabulary[token]] = count
    return result
