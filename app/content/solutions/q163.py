def q163(tree, alpha):
    """Prune a tree bottom-up using cost-complexity risk. Every node supplies leaf_error:
    the total training error if collapsed. Compare subtree error+alpha*leaves against
    leaf_error+alpha; return a copied pruned tree.
    """
    import copy

    tree = copy.deepcopy(tree)

    def prune(node):
        if "left" not in node:
            return (node["leaf_error"], 1)
        le, ln = prune(node["left"])
        re, rn = prune(node["right"])
        if node["leaf_error"] + alpha <= le + re + alpha * (ln + rn):
            node.pop("left")
            node.pop("right")
            node.pop("feature", None)
            node.pop("threshold", None)
            return (node["leaf_error"], 1)
        return (le + re, ln + rn)

    prune(tree)
    return tree
