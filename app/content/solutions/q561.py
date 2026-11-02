import numpy as np


def q561(examples, budget, pad_id=0):
    """Pack whole instruction examples into blocks; each example is
    (token_list,answer_mask,adapter_id). Return dicts with tokens, answer-only loss
    mask, adapter IDs, and allowed block-diagonal causal attention. Reject
    oversized/empty examples rather than truncating; padding has adapter -1 and no
    attention/loss. Mask denotes aligned target positions supplied by caller.
    """
    blocks = []
    current = []
    used = 0
    for tokens, answer, adapter in examples:
        if (
            not tokens
            or len(tokens) > budget
            or len(tokens) != len(answer)
            or (not any(answer))
        ):
            raise ValueError("invalid example")
        if used + len(tokens) > budget:
            blocks.append(current)
            current = []
            used = 0
        current.append((tokens, answer, adapter))
        used += len(tokens)
    if current:
        blocks.append(current)
    output = []
    for block in blocks:
        tokens = np.full(budget, pad_id)
        loss = np.zeros(budget, bool)
        adapters = np.full(budget, -1)
        segment = np.full(budget, -1)
        start = 0
        for i, (ids, mask, adapter) in enumerate(block):
            end = start + len(ids)
            tokens[start:end] = ids
            loss[start:end] = mask
            adapters[start:end] = adapter
            segment[start:end] = i
            start = end
        allowed = (
            (segment[:, None] == segment[None, :])
            & (segment[:, None] >= 0)
            & np.tri(budget, dtype=bool)
        )
        output.append(
            {
                "tokens": tokens,
                "loss_mask": loss,
                "adapter_ids": adapters,
                "attention": allowed,
            }
        )
    return output
