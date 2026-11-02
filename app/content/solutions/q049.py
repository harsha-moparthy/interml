def q049(model, frozen_prefixes, lr, weight_decay):
    """Freeze parameters whose names start with any supplied prefix. Return optimizer
    groups separating matrix weights (with decay) from biases and one-dimensional
    parameters (without decay).
    """
    decay = []
    no_decay = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(
            not any((name.startswith(prefix) for prefix in frozen_prefixes))
        )
        if parameter.requires_grad:
            (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    # Keep biases and normalization vectors out of weight decay.
    return [
        {"params": decay, "lr": lr, "weight_decay": weight_decay},
        {"params": no_decay, "lr": lr, "weight_decay": 0.0},
    ]
