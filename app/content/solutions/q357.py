def q357(input_width, output_width, rank, bias=False):
    """Return full linear-layer parameter count, adapter-only LoRA count, and adapter/full
    ratio. The optional base bias is counted only in the full layer.
    """
    full = input_width * output_width + (output_width if bias else 0)
    adapter = rank * (input_width + output_width)
    return (full, adapter, adapter / full)
