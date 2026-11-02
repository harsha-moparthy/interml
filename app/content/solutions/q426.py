def q426(
    parameters,
    tokens_per_second,
    gpus,
    peak_flops_per_gpu,
    flops_per_parameter_token=6,
):
    """Estimate model FLOPs utilization as k*N*global_tokens_per_second divided by
    aggregate GPU peak FLOPs/s. Return fraction, without clipping an inconsistent
    estimate.
    """
    return (
        flops_per_parameter_token
        * parameters
        * tokens_per_second
        / (gpus * peak_flops_per_gpu)
    )
