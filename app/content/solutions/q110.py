def q110(a, b, c):
    """Represent the scalar expression a*b+c as a dictionary of nodes. Each node contains
    an operation, parent IDs, and its evaluated value.
    """
    return {
        "a": ("input", (), a),
        "b": ("input", (), b),
        "c": ("input", (), c),
        "product": ("mul", ("a", "b"), a * b),
        "output": ("add", ("product", "c"), a * b + c),
    }
