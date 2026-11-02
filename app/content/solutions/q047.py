def q047(module, x):
    """Capture a tensor-valued module's output using a forward hook. Return a detached
    clone and always remove the hook.
    """
    captured = []
    handle = module.register_forward_hook(
        lambda m, args, y: captured.append(y.detach().clone())
    )
    # Remove the hook even if the model raises; leaked hooks duplicate later captures.
    try:
        module(x)
    finally:
        handle.remove()
    return captured[-1]
