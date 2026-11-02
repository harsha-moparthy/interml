from collections import Counter


def merge_pair(symbols, pair):
    """Replace nonoverlapping occurrences from left to right."""
    output = []
    index = 0
    while index < len(symbols):
        if index + 1 < len(symbols) and tuple(symbols[index : index + 2]) == pair:
            output.append(symbols[index] + symbols[index + 1])
            index += 2
        else:
            output.append(symbols[index])
            index += 1
    return output


def train_bpe(words, num_merges=100):
    """Learn character BPE with a terminal marker; repeated words retain frequency.

    Ties are lexicographic. Input words cannot contain the reserved marker.
    This educational tokenizer has no byte fallback, normalization, or offsets.
    """
    if num_merges < 0 or any("</w>" in word for word in words):
        raise ValueError("invalid merge count or reserved marker in input")
    sequences = [list(word) + ["</w>"] for word in words]
    merges = []
    for _ in range(num_merges):
        counts = Counter(
            pair for symbols in sequences for pair in zip(symbols, symbols[1:])
        )
        if not counts:
            break
        pair = min(counts, key=lambda item: (-counts[item], item))
        merges.append(pair)
        sequences = [merge_pair(symbols, pair) for symbols in sequences]
    return merges


def encode_bpe(word, merges):
    """Apply learned merges in training rank order, including the terminal marker."""
    if "</w>" in word:
        raise ValueError("input contains reserved terminal marker")
    symbols = list(word) + ["</w>"]
    for pair in merges:
        symbols = merge_pair(symbols, pair)
    return symbols
