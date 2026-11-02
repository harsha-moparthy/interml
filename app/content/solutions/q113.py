def q113(nodes, output):
    """Run scalar reverse-mode autodiff on a topologically ordered list of
    (name,operation,parent_names,value). Support input, add, multiply, and tanh; return
    gradients by name.
    """
    values = {name: value for name, op, parents, value in nodes}
    gradient = {name: 0.0 for name in values}
    gradient[output] = 1.0
    for name, op, parents, value in reversed(nodes):
        g = gradient[name]
        if op == "input":
            continue
        if op == "add":
            local = [1.0, 1.0]
        elif op == "mul":
            local = [values[parents[1]], values[parents[0]]]
        elif op == "tanh":
            local = [1 - value * value]
        else:
            raise ValueError("unknown operation")
        for parent, multiplier in zip(parents, local):
            gradient[parent] += g * multiplier
    return gradient
