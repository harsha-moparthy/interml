import torch


def quantize_int8(weight):
    """Per-tensor asymmetric signed INT8 quantization; return codes, scale, zero point.

    Include zero in the calibration range, so all-positive/all-negative constants
    remain representable. Dequantization is (codes.float()-zero_point)*scale.
    """
    if not weight.numel() or not torch.isfinite(weight).all():
        raise ValueError("expected finite nonempty weights")
    values = weight.float()
    minimum = min(float(values.min()), 0.0)
    maximum = max(float(values.max()), 0.0)
    scale = (maximum - minimum) / 255 if maximum != minimum else 1.0
    zero_point = max(-128, min(127, round(-128 - minimum / scale)))
    codes = (values / scale + zero_point).round().clamp(-128, 127).to(torch.int8)
    return codes, scale, zero_point
