import math

import torch


def scaled_dot_product_attention(
    query, key, value, *, mask=None, causal=False, query_offset=0
):
    """Attend from (B,H,L,D) queries to (B,H,S,D) keys and values.

    mask is boolean, broadcastable to (B,H,L,S); True means allowed.
    Use shape (B,1,1,S) for a key-padding mask. Fully masked rows return zero.
    query_offset is the number of cached tokens before this query block.
    Inputs must be finite floating tensors on the same device and dtype.
    Time: O(B*H*L*S*D); score storage: O(B*H*L*S).
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have four dimensions")
    if query.shape[:2] != key.shape[:2] or key.shape[:3] != value.shape[:3]:
        raise ValueError("batch/head dimensions and key/value lengths must match")
    if query.shape[-1] != key.shape[-1] or query.shape[-1] == 0:
        raise ValueError("query and key must have the same positive head width")
    if key.shape[-2] == 0 or query_offset < 0:
        raise ValueError("keys must be nonempty and query_offset nonnegative")

    # Keep score accumulation in float32 for low-precision inputs.
    work_dtype = (
        torch.float32 if query.dtype in (torch.float16, torch.bfloat16) else query.dtype
    )
    scores = query.to(work_dtype) @ key.to(work_dtype).transpose(-2, -1)
    scores = scores / math.sqrt(query.shape[-1])
    allowed = torch.ones(scores.shape[-2:], dtype=torch.bool, device=query.device)
    if causal:
        query_positions = query_offset + torch.arange(
            query.shape[-2], device=query.device
        )
        key_positions = torch.arange(key.shape[-2], device=query.device)
        allowed = key_positions[None, :] <= query_positions[:, None]
    if mask is not None:
        if mask.dtype != torch.bool:
            raise ValueError("mask must be boolean; True means allowed")
        allowed = allowed & mask

    scores = scores.masked_fill(~allowed, -torch.inf)
    has_key = allowed.any(dim=-1, keepdim=True)
    # Avoid softmax(-inf, ..., -inf), including NaNs in its backward pass.
    safe_scores = torch.where(has_key, scores, torch.zeros_like(scores))
    weights = torch.softmax(safe_scores, dim=-1).masked_fill(~allowed, 0)
    return (weights @ value.to(work_dtype)).to(value.dtype)


class MultiHeadSelfAttention(torch.nn.Module):
    """Fused QKV attention for (batch, sequence, model_dim) inputs."""

    def __init__(self, model_dim, num_heads):
        super().__init__()
        if num_heads <= 0 or model_dim <= 0 or model_dim % num_heads:
            raise ValueError("model_dim must be positive and divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.qkv = torch.nn.Linear(model_dim, 3 * model_dim)
        self.output = torch.nn.Linear(model_dim, model_dim)

    def forward(self, x, *, mask=None, causal=False):
        batch, length, width = x.shape
        # One projection, then split the head axis; no per-head Python loop.
        projected = self.qkv(x).reshape(batch, length, 3, self.num_heads, self.head_dim)
        query, key, value = projected.permute(2, 0, 3, 1, 4).unbind(0)
        context = scaled_dot_product_attention(
            query, key, value, mask=mask, causal=causal
        )
        context = context.transpose(1, 2).reshape(batch, length, width)
        return self.output(context)
