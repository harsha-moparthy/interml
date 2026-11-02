import torch


def q416(x, block_size=128):
    """Quantize a floating Torch tensor blockwise to E4M3FN FP8 with per-block scaling to
    maximum finite magnitude 448. Return FP8 blocks, float32 scales, and reconstruction
    in the original shape.
    """
    if block_size < 1:
        raise ValueError("positive block size required")
    flat = x.float().flatten()
    codes = []
    scales = []
    rebuilt = []
    for block in flat.split(block_size):
        scale = block.abs().amax().clamp_min(1e-12) / 448
        code = (block / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        codes.append(code)
        scales.append(scale)
        rebuilt.append(code.float() * scale)
    return (codes, torch.stack(scales), torch.cat(rebuilt).reshape(x.shape))
