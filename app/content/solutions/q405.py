import numpy as np


def q405(arrival_time, token_timestamps):
    """Compute TTFT, individual inter-token latencies, and decode tokens/second as
    (N-1)/(last-first). Require ordered timestamps no earlier than arrival; rate is NaN
    for fewer than two tokens or zero duration.
    """
    t = np.asarray(token_timestamps, float)
    if not len(t) or t[0] < arrival_time or np.any(np.diff(t) < 0):
        raise ValueError("invalid timestamps")
    return {
        "ttft": float(t[0] - arrival_time),
        "itl": np.diff(t),
        "decode_tps": (
            float((len(t) - 1) / (t[-1] - t[0]))
            if len(t) > 1 and t[-1] > t[0]
            else float("nan")
        ),
    }
