def q112(parents, output):
    """Topologically order all nodes reachable from output in a parent-list computation
    graph. Detect cycles; place parents before children.
    """
    state = {}
    order = []

    def visit(node):
        if state.get(node) == 1:
            raise ValueError("cycle")
        if state.get(node) == 2:
            return
        state[node] = 1
        for parent in parents.get(node, []):
            visit(parent)
        state[node] = 2
        order.append(node)

    visit(output)
    return order
