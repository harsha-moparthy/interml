import torch


def q344(x, router, experts, shared, k=2):
    """Apply a sparse MoE with a shared expert. Router(x) returns logits; experts and
    shared are callable token-wise maps. Softmax only the selected top-k logits and add
    the shared expert output.
    """
    logits = router(x)
    scores, indices = torch.topk(logits, k, dim=-1)
    weights = torch.softmax(scores, dim=-1)
    result = shared(x)
    for expert_id, expert in enumerate(experts):
        token, slot = torch.where(indices == expert_id)
        if len(token):
            result = result.index_add(
                0, token, expert(x[token]) * weights[token, slot, None]
            )
    return result
