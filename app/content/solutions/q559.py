import torch


def q559(base, expert_outputs, gate_weights):
    """Return base[B,T,D] + sum_e gates[B,T,E]*expert_outputs[B,T,E,D]. Gates must be
    nonnegative and sum to one per token. Zero/one-hot gates express sparse/hard
    routing; this reference evaluates supplied expert outputs rather than scheduling
    expert computation.
    """
    if (
        expert_outputs.shape[:-1] != gate_weights.shape
        or expert_outputs.shape[:2] + expert_outputs.shape[-1:] != base.shape
    ):
        raise ValueError("shape mismatch")
    if torch.any(gate_weights < 0) or not torch.allclose(
        gate_weights.sum(-1), torch.ones_like(gate_weights[..., 0])
    ):
        raise ValueError("invalid gates")
    return base + (expert_outputs * gate_weights[..., None]).sum(-2)
