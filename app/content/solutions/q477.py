def q477(unconditional, conditional, guidance_scale):
    """Combine unconditional and conditional model predictions for classifier-free
    guidance. Scale 0 yields unconditional, scale 1 yields conditional.
    """
    return unconditional + guidance_scale * (conditional - unconditional)
