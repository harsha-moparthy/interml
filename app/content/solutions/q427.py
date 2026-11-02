def q427(flops, bytes_moved, peak_flops, bandwidth):
    """Apply the roofline model: compute arithmetic intensity, attainable FLOPs/s, ideal
    runtime lower bound, and the limiting resource. Assume positive byte traffic and
    hardware rates.
    """
    intensity = flops / bytes_moved
    achieved = min(peak_flops, intensity * bandwidth)
    return {
        "intensity": intensity,
        "attainable_flops": achieved,
        "seconds": max(flops / peak_flops, bytes_moved / bandwidth),
        "bound": (
            "compute" if flops / peak_flops >= bytes_moved / bandwidth else "memory"
        ),
    }
