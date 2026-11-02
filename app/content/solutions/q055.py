import torch


def q055(model, x):
    """Perform inference under autocast: use float16 on CUDA and bfloat16 on CPU. Return
    the output tensor with gradients disabled.
    """
    dtype = torch.float16 if x.device.type == "cuda" else torch.bfloat16
    with torch.no_grad(), torch.autocast(device_type=x.device.type, dtype=dtype):
        return model(x)
