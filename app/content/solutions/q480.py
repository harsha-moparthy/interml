from torch.nn import functional as F


def q480(model, noise, data, times):
    """Compute rectified-flow velocity-matching Torch loss using x_t=(1-t)*noise+t*data and
    target data-noise. times is (batch,), with values in [0,1]; model accepts
    (x_t,times).
    """
    t = times.reshape((-1,) + (1,) * (data.ndim - 1))
    path = (1 - t) * noise + t * data
    return F.mse_loss(model(path, times), data - noise)
