from collections import Counter


def q294(words, merges=10):
    """Train a character-level BPE merge table on a list of words. Append an end-of-word
    marker; repeatedly merge the most frequent adjacent pair, breaking ties
    lexicographically. Return merge pairs and segmented training words.
    """
    sequences = [list(word) + ["</w>"] for word in words]
    learned = []
    for _ in range(merges):
        counts = Counter(
            (pair for sequence in sequences for pair in zip(sequence, sequence[1:]))
        )
        if not counts:
            break
        # Lexicographic tie-breaking makes training reproducible when pair counts match.
        pair = min(counts, key=lambda p: (-counts[p], p))
        learned.append(pair)
        next_sequences = []
        for sequence in sequences:
            result = []
            position = 0
            while position < len(sequence):
                if (
                    position + 1 < len(sequence)
                    and (sequence[position], sequence[position + 1]) == pair
                ):
                    result.append(sequence[position] + sequence[position + 1])
                    position += 2
                else:
                    result.append(sequence[position])
                    position += 1
            next_sequences.append(result)
        sequences = next_sequences
    return (learned, sequences)
