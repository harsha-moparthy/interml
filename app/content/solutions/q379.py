import math
from collections import Counter


def q379(n, correct, k, answers):
    """Return the unbiased pass@k estimate for n samples with correct successes, plus the
    most frequent hashable answer. Require 1<=k<=n and 0<=correct<=n; ties follow first
    occurrence.
    """
    if not 1 <= k <= n or not 0 <= correct <= n:
        raise ValueError("invalid counts")
    failure = 0 if n - correct < k else math.comb(n - correct, k) / math.comb(n, k)
    majority = Counter(answers).most_common(1)
    # The estimator counts the probability that a sample of k contains at least one success.
    return (1 - failure, majority[0][0] if majority else None)
