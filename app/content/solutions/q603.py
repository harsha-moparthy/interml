import math

import torch


def grouped_query_attention(query, key, value, *, mask=None, causal=False):
    """GQA for Q(B,Hq,L,D), K/V(B,Hkv,S,D); Hkv=1 gives MQA.

    True in mask means allowed; mask must broadcast to (B,Hq,L,S).
    K/V heads are broadcast, never repeat_interleaved into a full copy.
    Score storage still costs O(B*Hq*L*S); this is not a fused kernel.
    """
    batch, query_heads, length, width = query.shape
    kv_heads = key.shape[1]
    if kv_heads <= 0 or query_heads % kv_heads or width <= 0:
        raise ValueError("positive KV heads must divide query heads")
    if key.shape != value.shape or key.shape[0] != batch or key.shape[-1] != width:
        raise ValueError("K/V must match and share Q's batch and head width")
    if key.shape[-2] == 0:
        raise ValueError("keys must be nonempty")
    groups = query_heads // kv_heads
    work_dtype = (
        torch.float32 if query.dtype in (torch.float16, torch.bfloat16) else query.dtype
    )
    grouped_queries = query.to(work_dtype).reshape(
        batch, kv_heads, groups, length, width
    )
    scores = grouped_queries @ key.to(work_dtype)[:, :, None].transpose(-2, -1)
    scores = scores.reshape(batch, query_heads, length, key.shape[-2]) / math.sqrt(
        width
    )
    allowed = torch.ones(scores.shape[-2:], dtype=torch.bool, device=query.device)
    if causal:
        allowed = allowed.tril()
    if mask is not None:
        if mask.dtype != torch.bool:
            raise ValueError("mask must be boolean")
        allowed = allowed & mask
    scores = scores.masked_fill(~allowed, -torch.inf)
    scores = torch.where(
        allowed.any(-1, keepdim=True), scores, torch.zeros_like(scores)
    )
    weights = scores.softmax(-1).masked_fill(~allowed, 0)
    weights = weights.reshape(batch, kv_heads, groups, length, key.shape[-2])
    context = weights @ value.to(work_dtype)[:, :, None]
    return context.reshape(batch, query_heads, length, width).to(value.dtype)
