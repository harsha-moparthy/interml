import numpy as np


def q298(words, vocabulary, boundary="."):
    """Count character bigrams within words, adding a supplied boundary token at both ends.
    Vocabulary must include every character and the boundary.
    """
    result = np.zeros((len(vocabulary), len(vocabulary)), int)
    for word in words:
        tokens = [boundary] + list(word) + [boundary]
        for a, b in zip(tokens, tokens[1:]):
            result[vocabulary[a], vocabulary[b]] += 1
    return result
