def q307(training_tokens, benchmark_tokens, n=5):
    """Measure benchmark contamination as the fraction of distinct benchmark n-grams
    appearing in a training token sequence. Return zero if the benchmark has no complete
    n-gram.
    """
    if n <= 0:
        raise ValueError("positive n required")

    def grams(tokens):
        return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}

    train, test = (grams(training_tokens), grams(benchmark_tokens))
    return len(train & test) / len(test) if test else 0.0
