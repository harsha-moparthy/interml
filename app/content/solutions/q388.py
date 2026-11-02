import numpy as np
from scipy.special import logsumexp


def q388(next_logits, prompt, beam_width, max_new_tokens, eos_id):
    """Implement beam search ranked by cumulative log probability without length
    normalization. Retain completed beams unchanged, and return the highest-scoring
    token sequence and score.
    """
    if beam_width < 1:
        raise ValueError("positive beam width required")
    beams = [(list(prompt), 0.0, False)]
    for _ in range(max_new_tokens):
        candidates = []
        for tokens, score, done in beams:
            if done:
                candidates.append((tokens, score, True))
                continue
            logits = np.asarray(next_logits(tokens), float)
            logp = logits - logsumexp(logits)
            for token in np.argsort(-logp, kind="stable")[:beam_width]:
                candidates.append(
                    (
                        tokens + [int(token)],
                        score + float(logp[token]),
                        token == eos_id,
                    )
                )
        beams = sorted(candidates, key=lambda row: row[1], reverse=True)[:beam_width]
        if all((row[2] for row in beams)):
            break
    return beams[0][:2]
