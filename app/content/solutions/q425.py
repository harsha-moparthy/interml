import numpy as np


def q425(world_size, data, tensor, pipeline, context):
    """Map global ranks to a Cartesian (data,tensor,pipeline,context) topology. Require the
    product to equal world size; context varies fastest. Return coordinates and rank
    groups for each communication axis.
    """
    shape = (data, tensor, pipeline, context)
    if min(shape) < 1 or np.prod(shape) != world_size:
        raise ValueError("invalid factorization")
    ranks = np.arange(world_size).reshape(shape)
    coordinates = [np.unravel_index(i, shape) for i in range(world_size)]
    groups = {
        name: np.moveaxis(ranks, axis, -1).reshape(-1, shape[axis]).tolist()
        for axis, name in enumerate(("data", "tensor", "pipeline", "context"))
    }
    return (coordinates, groups)
