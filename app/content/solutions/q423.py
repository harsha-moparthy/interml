def q423(stages, microbatches, stage_seconds=1):
    """For balanced GPipe fill/drain with equal forward/backward stage times, return bubble
    fraction (p-1)/(m+p-1), utilization, and total forward-plus-backward time.
    """
    if min(stages, microbatches) < 1:
        raise ValueError("positive counts required")
    bubble = (stages - 1) / (microbatches + stages - 1)
    return (
        bubble,
        1 - bubble,
        2 * (microbatches + stages - 1) * stage_seconds,
    )
