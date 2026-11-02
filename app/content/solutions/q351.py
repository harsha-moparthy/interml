import torch


def q351(examples, pad_id=0, ignore_index=-100):
    """Collate instruction examples containing prompt and response token lists. Create
    next-token inputs/labels, pad on the right, and supervise response tokens only.
    Return tensors and a valid-input mask.
    """
    sequences = [e["prompt"] + e["response"] for e in examples]
    length = max((len(s) - 1 for s in sequences))
    inputs = torch.full((len(examples), length), pad_id, dtype=torch.long)
    labels = torch.full_like(inputs, ignore_index)
    mask = torch.zeros_like(inputs, dtype=torch.bool)
    for i, (example, sequence) in enumerate(zip(examples, sequences)):
        n = len(sequence) - 1
        inputs[i, :n] = torch.tensor(sequence[:-1])
        mask[i, :n] = True
        target = torch.tensor(sequence[1:])
        start = max(len(example["prompt"]) - 1, 0)
        labels[i, start:n] = target[start:]
    return (inputs, labels, mask)
