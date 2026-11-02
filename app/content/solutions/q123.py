import pandas as pd


def q123(categories):
    """Return a dictionary of relative category frequencies for a nonempty sequence of
    hashable, nonmissing categories.
    """
    return pd.Series(categories).value_counts(normalize=True).to_dict()
