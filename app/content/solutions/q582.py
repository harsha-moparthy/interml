import numpy as np


def q582(predictions, targets, patients, ignore=-1):
    """Aggregate binary segmentation arrays [N,...] by patient and globally. Predictions
    are Boolean or 0/1; ignored target pixels are omitted. Return micro IoU/Dice and
    mean patient IoU/Dice. For both-empty masks define both metrics as one; a patient
    with only ignored pixels is excluded.
    """
    p = np.asarray(predictions, bool)
    y = np.asarray(targets)
    if p.shape != y.shape or len(patients) != len(y):
        raise ValueError("shape/patient mismatch")
    counts = {}
    for pred, truth, patient in zip(p, y, patients):
        valid = truth != ignore
        if not valid.any():
            continue
        a = pred[valid]
        b = truth[valid] != 0
        row = np.array([(a & b).sum(), (a & ~b).sum(), (~a & b).sum()], float)
        counts[patient] = counts.get(patient, np.zeros(3)) + row

    def metric(c):
        tp, fp, fn = c
        union = tp + fp + fn
        denom = 2 * tp + fp + fn
        return (
            tp / union if union else 1.0,
            2 * tp / denom if denom else 1.0,
        )

    if not counts:
        return ((np.nan, np.nan), (np.nan, np.nan))
    return (
        metric(sum(counts.values())),
        tuple(np.mean([metric(c) for c in counts.values()], axis=0)),
    )
