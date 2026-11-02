import torch


def apply_rope(query, key, positions, *, base=10000.0):
    """Rotate adjacent coordinate pairs in Q/K(B,H,L,D), preserving norms.

    positions has shape (L,); use absolute positions when decoding with a cache.
    Both tensors must share device, dtype, sequence length, and even head width.
    Head counts may differ for grouped-query attention. Time/storage: O(B*H*L*D).
    """
    width = query.shape[-1]
    if width == 0 or width % 2 or base <= 0:
        raise ValueError("head width must be positive and even; base must be positive")
    if query.shape[-2:] != key.shape[-2:]:
        raise ValueError("query/key sequence lengths and head widths must match")
    positions = torch.as_tensor(positions, device=query.device)
    if positions.shape != (query.shape[-2],):
        raise ValueError("positions must contain one position per token")
    work_dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
    frequencies = base ** (
        -torch.arange(0, width, 2, device=query.device, dtype=work_dtype) / width
    )
    angles = positions.to(work_dtype)[:, None] * frequencies
    cosine, sine = angles.cos(), angles.sin()

    def rotate(x):
        even, odd = x[..., 0::2].to(work_dtype), x[..., 1::2].to(work_dtype)
        # Interleave rotated pairs without writing into the input tensor.
        pairs = torch.stack(
            (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
        )
        return pairs.flatten(-2).to(x.dtype)

    return rotate(query), rotate(key)
