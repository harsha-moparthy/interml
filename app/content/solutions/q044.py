from torch import nn


def q044(dimensions):
    """Build an MLP from a list of layer widths. Insert ReLU between affine layers, leaving
    the last layer linear.
    """
    layers = []
    for layer_index, (input_width, output_width) in enumerate(
        zip(dimensions, dimensions[1:])
    ):
        layers.append(nn.Linear(input_width, output_width))
        if layer_index < len(dimensions) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)
