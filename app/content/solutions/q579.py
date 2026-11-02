import numpy as np


def q579(patient_ids, train_fraction=0.7, validation_fraction=0.15, seed=0):
    """Assign every row from a patient to one deterministic train/validation/test split.
    Shuffle sorted unique patient IDs; floor patient counts for train and validation,
    remainder test. Return row split labels and patient mapping. Augmented rows must
    retain their original patient IDs.
    """
    if (
        min(train_fraction, validation_fraction) < 0
        or train_fraction + validation_fraction > 1
    ):
        raise ValueError("invalid fractions")
    patients = np.array(sorted(set(patient_ids)), dtype=object)
    np.random.default_rng(seed).shuffle(patients)
    a = int(len(patients) * train_fraction)
    b = a + int(len(patients) * validation_fraction)
    mapping = {
        p: "train" if i < a else "validation" if i < b else "test"
        for i, p in enumerate(patients)
    }
    return ([mapping[p] for p in patient_ids], mapping)
