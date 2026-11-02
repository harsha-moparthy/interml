import numpy as np


def q398(target_probs, draft_probs, prompt, steps, draft_length, seed=0):
    """Simulate exact speculative sampling with supplied normalized probability callbacks
    on token histories. Accept draft x with min(1,p[x]/q[x]); rejection samples
    normalized max(p-q,0). Each outer step emits up to draft_length+1 tokens.
    """
    rng = np.random.default_rng(seed)
    tokens = list(prompt)
    # Acceptance uses p/q; a rejected proposal is replaced from the positive residual p-q.
    for _ in range(steps):
        drafts = []
        draft_distributions = []
        prefix = tokens.copy()
        for _ in range(draft_length):
            draft_distribution = np.asarray(draft_probs(prefix), float)
            token = int(rng.choice(len(draft_distribution), p=draft_distribution))
            drafts.append(token)
            draft_distributions.append(draft_distribution)
            prefix.append(token)
        rejected = False
        for token, draft_distribution in zip(drafts, draft_distributions):
            target_distribution = np.asarray(target_probs(tokens), float)
            if rng.random() < min(
                1.0, target_distribution[token] / draft_distribution[token]
            ):
                tokens.append(token)
            else:
                residual = np.maximum(target_distribution - draft_distribution, 0)
                residual /= residual.sum()
                tokens.append(int(rng.choice(len(target_distribution), p=residual)))
                rejected = True
                break
        if not rejected:
            target_distribution = np.asarray(target_probs(tokens), float)
            tokens.append(
                int(rng.choice(len(target_distribution), p=target_distribution))
            )
    return tokens
