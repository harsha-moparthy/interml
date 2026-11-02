def q050(features, labels, batch_size):
    """Create a deterministic DataLoader over paired tensors without shuffling or
    discarding the final partial batch.
    """
    from torch.utils.data import TensorDataset, DataLoader

    return DataLoader(
        TensorDataset(features, labels),
        batch_size=batch_size,
        shuffle=False,
    )
