def q563(
    parameters,
    world_size,
    stage,
    weight_bytes=2,
    gradient_bytes=2,
    master_bytes=4,
    moment_bytes=8,
):
    """Estimate per-rank MODEL STATE bytes under ZeRO stages 0-3. Stage1 shards FP32 master
    copy and optimizer moments; stage2 additionally shards gradients; stage3 also shards
    weights. Return breakdown and total using ideal equal partitioning. Excludes
    activations, gathers, buffers, padding, and fragmentation; it is not peak memory.
    """
    if parameters < 0 or world_size < 1 or stage not in (0, 1, 2, 3):
        raise ValueError("invalid configuration")
    result = {
        "weights": parameters * weight_bytes / (world_size if stage >= 3 else 1),
        "gradients": parameters * gradient_bytes / (world_size if stage >= 2 else 1),
        "master": parameters * master_bytes / (world_size if stage >= 1 else 1),
        "moments": parameters * moment_bytes / (world_size if stage >= 1 else 1),
    }
    result["total"] = sum(result.values())
    return result
