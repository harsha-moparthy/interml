import numpy as np


def q463(
    initial_state,
    actions,
    transition,
    terminal_value,
    rollout,
    simulations=100,
    exploration=1.414,
    seed=0,
):
    """Implement UCT MCTS for a single-agent finite tree. Callbacks provide legal actions,
    deterministic next states, terminal value or None, and a rollout value. All rewards
    use the same root perspective; return best root action and visit counts.
    """
    rng = np.random.default_rng(seed)

    def node(state):
        return {
            "state": state,
            "n": 0,
            "w": 0.0,
            "children": {},
            "untried": list(actions(state)),
        }

    root = node(initial_state)
    for _ in range(simulations):
        current = root
        path = [root]
        while (
            terminal_value(current["state"]) is None
            and (not current["untried"])
            and current["children"]
        ):
            current = max(
                current["children"].values(),
                key=lambda child: child["w"] / child["n"]
                + exploration * np.sqrt(np.log(max(1, current["n"])) / child["n"]),
            )
            path.append(current)
        value = terminal_value(current["state"])
        if value is None and current["untried"]:
            index = int(rng.integers(len(current["untried"])))
            action = current["untried"].pop(index)
            child = node(transition(current["state"], action))
            current["children"][action] = child
            current = child
            path.append(child)
            value = terminal_value(current["state"])
        if value is None:
            value = rollout(current["state"], rng)
        for item in path:
            item["n"] += 1
            item["w"] += value
    visits = {a: c["n"] for a, c in root["children"].items()}
    return (max(visits, key=visits.get) if visits else None, visits)
