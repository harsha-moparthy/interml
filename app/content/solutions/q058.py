import torch


def q058(shadow, model, decay):
    """Update an EMA model in place: average parameters and copy buffers from the live
    model. Models must have identical structure.
    """
    if not 0 <= decay <= 1:
        raise ValueError("invalid decay")
    with torch.no_grad():
        source = dict(model.named_parameters())
        for name, parameter in shadow.named_parameters():
            parameter.lerp_(source[name], 1 - decay)
        # Buffers include running statistics and counters; copy rather than average them.
        buffers = dict(model.named_buffers())
        for name, buffer in shadow.named_buffers():
            buffer.copy_(buffers[name])
    return shadow
