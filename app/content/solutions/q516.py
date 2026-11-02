import torch
from torch.nn import functional as F


def q516(
    extractor,
    prediction,
    target,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    """Compute feature L1 loss using a supplied single-output frozen extractor on RGB BCHW
    images. Normalize with supplied channel means/stds. Disable extractor parameter
    gradients but retain the prediction path; detach the target path. No pretrained
    weights are downloaded.
    """
    extractor.eval().requires_grad_(False)
    mu = prediction.new_tensor(mean)[None, :, None, None]
    sigma = prediction.new_tensor(std)[None, :, None, None]
    pred_features = extractor((prediction - mu) / sigma)
    with torch.no_grad():
        truth = extractor((target - mu) / sigma)
    return F.l1_loss(pred_features, truth)
