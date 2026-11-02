import torch


def sample_tokens(logits, *, temperature=1.0, top_k=None, top_p=1.0, generator=None):
    """Sample one token per (batch, vocabulary) row after top-k then nucleus filtering.

    Inputs are finite logits. temperature=0 selects argmax. Keep the token that
    crosses the top-p boundary, so the retained mass reaches the threshold.
    Sorting costs O(B*V*log(V)); filtering uses O(B*V) storage.
    """
    if logits.ndim != 2 or logits.shape[1] == 0 or not torch.isfinite(logits).all():
        raise ValueError("logits must be finite with a nonempty vocabulary")
    if temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("temperature must be nonnegative and top_p in (0,1]")
    if top_k is not None and not 1 <= top_k <= logits.shape[-1]:
        raise ValueError("top_k is outside the vocabulary")
    if temperature == 0:
        return logits.argmax(-1)
    scores, indices = (logits.float() / temperature).sort(dim=-1, descending=True)
    if top_k is not None:
        scores[:, top_k:] = -torch.inf
    probabilities = scores.softmax(-1)
    # Mass strictly before each token decides whether it survives.
    remove = probabilities.cumsum(-1) - probabilities >= top_p
    scores = scores.masked_fill(remove, -torch.inf)
    selected = torch.multinomial(scores.softmax(-1), 1, generator=generator)
    return indices.gather(-1, selected).squeeze(-1)
