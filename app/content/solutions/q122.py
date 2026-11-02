import pandas as pd


def q122(categories, targets, smoothing=10):
    """Fit a smoothed target-encoding dictionary: each category mean is shrunk toward the
    global mean by smoothing pseudo-observations. Return the mapping and fallback global
    mean. Fit only on the training fold.
    """
    frame = pd.DataFrame({"key": categories, "y": targets})
    # Smooth rare categories toward the training-set mean; fit only on training data.
    global_mean = float(frame.y.mean())
    stats = frame.groupby("key").y.agg(["sum", "count"])
    values = (stats["sum"] + smoothing * global_mean) / (stats["count"] + smoothing)
    return (values.to_dict(), global_mean)
