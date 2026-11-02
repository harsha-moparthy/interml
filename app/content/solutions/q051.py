import torch


def q051(sequences, pad_id=0):
    """Collate nonempty lists of integer token sequences into padded long tensors and a
    Boolean valid-token mask. Padding is on the right.
    """
    lengths = torch.tensor([len(sequence) for sequence in sequences])
    width = int(lengths.max()) if len(lengths) else 0
    batch = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    for row, sequence in enumerate(sequences):
        batch[row, : len(sequence)] = torch.as_tensor(sequence, dtype=torch.long)
    return (batch, torch.arange(width)[None, :] < lengths[:, None])
