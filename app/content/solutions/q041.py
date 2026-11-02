def q041(x, shape, dim0, dim1):
    """Reshape a tensor, transpose two dimensions, and return contiguous storage. Element
    count must be preserved.
    """
    return x.reshape(shape).transpose(dim0, dim1).contiguous()
