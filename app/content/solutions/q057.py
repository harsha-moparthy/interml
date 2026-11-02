def q057(block, x):
    """Evaluate a differentiable block with non-reentrant activation checkpointing,
    preserving random state for recomputation.
    """
    from torch.utils.checkpoint import checkpoint

    return checkpoint(block, x, use_reentrant=False, preserve_rng_state=True)
