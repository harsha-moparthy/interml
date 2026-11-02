import numpy as np


def q209(scores, targets):
    """Fit nondecreasing isotonic calibration with the pool-adjacent-violators algorithm.
    Group tied scores before fitting; return unique sorted scores and calibrated values.
    """
    scores, targets = (
        np.asarray(scores, float),
        np.asarray(targets, float),
    )
    unique, inverse = np.unique(scores, return_inverse=True)
    weight = np.bincount(inverse)
    total = np.bincount(inverse, weights=targets)
    # Pool adjacent violating blocks until their weighted means are nondecreasing.
    blocks = []
    for i, (weight_sum, target_sum) in enumerate(zip(weight, total)):
        blocks.append([i, i, float(weight_sum), float(target_sum)])
        while (
            len(blocks) > 1
            and blocks[-2][3] / blocks[-2][2] > blocks[-1][3] / blocks[-1][2]
        ):
            right_block = blocks.pop()
            left_block = blocks.pop()
            blocks.append(
                [
                    left_block[0],
                    right_block[1],
                    left_block[2] + right_block[2],
                    left_block[3] + right_block[3],
                ]
            )
    result = np.empty(len(unique))
    for start, end, weight_sum, target_sum in blocks:
        result[start : end + 1] = target_sum / weight_sum
    return (unique, result)
