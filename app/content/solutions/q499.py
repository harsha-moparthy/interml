import numpy as np


def q499(feature_times, label_end_times, test_blocks, gap=0):
    """Build expanding walk-forward splits for test index blocks (start,stop). Train only
    on earlier rows whose label end time is strictly before first test feature time
    minus gap; all times share numeric units and are sorted.
    """
    times = np.asarray(feature_times)
    ends = np.asarray(label_end_times)
    splits = []
    # Purge the gap before validation so overlapping feature/label windows cannot leak.
    for start, stop in test_blocks:
        train = np.flatnonzero(
            (np.arange(len(times)) < start) & (ends < times[start] - gap)
        )
        splits.append((train, np.arange(start, stop)))
    return splits
