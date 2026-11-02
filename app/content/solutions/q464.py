import numpy as np


def q464(priors, value_sums, visits, c_puct=1.5):
    """Select a PUCT action from root-perspective value sums and visits:
    Q+c*P*sqrt(max(1,total_visits))/(1+visits). Unvisited Q values are zero.
    """
    n = np.asarray(visits)
    q = np.divide(value_sums, n, out=np.zeros_like(n, dtype=float), where=n > 0)
    score = q + c_puct * np.asarray(priors) * np.sqrt(max(1, n.sum())) / (1 + n)
    return (int(score.argmax()), score)
