import torch


def q040(values):
    """Create an independent float32 CPU tensor from a Python sequence."""
    return torch.tensor(values, dtype=torch.float32)
