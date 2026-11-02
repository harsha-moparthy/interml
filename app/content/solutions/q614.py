import numpy as np


def verify_draft(tokens, draft_probabilities, target_probabilities, *, seed=0):
    """Verify K proposed tokens using q(K,V) and p(K+1,V).

    p[i] must be conditioned on the prefix plus the first i proposals. The last
    row supplies the bonus token if every proposal is accepted. Return emitted
    tokens and accepted count. On rejection, stop: later rows have stale prefixes.
    """
    rng = np.random.default_rng(seed)
    q = np.asarray(draft_probabilities, dtype=float)
    p = np.asarray(target_probabilities, dtype=float)
    tokens = np.asarray(tokens, dtype=int)
    if q.ndim != 2 or p.shape != (len(q) + 1, q.shape[1]) or tokens.shape != (len(q),):
        raise ValueError("expected q(K,V), p(K+1,V), and K proposals")
    for probabilities in (p, q):
        if (
            not np.isfinite(probabilities).all()
            or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(-1), 1)
        ):
            raise ValueError("rows must be normalized probability distributions")
    if np.any((tokens < 0) | (tokens >= q.shape[1])):
        raise ValueError("proposal outside vocabulary")
    output = []
    for index, token in enumerate(tokens):
        if q[index, token] <= 0:
            raise ValueError("a proposal must have positive draft probability")
        if rng.random() < min(1.0, p[index, token] / q[index, token]):
            output.append(int(token))
            continue
        residual = np.maximum(p[index] - q[index], 0)
        residual /= residual.sum()
        output.append(int(rng.choice(len(residual), p=residual)))
        return output, index
    output.append(int(rng.choice(p.shape[1], p=p[-1])))
    return output, len(tokens)
