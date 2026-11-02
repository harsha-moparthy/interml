import pandas as pd


def q014(left_categories, right_categories):
    """Count co-occurrences of paired categories in a contingency table."""
    return pd.crosstab(
        pd.Series(left_categories, name="left"),
        pd.Series(right_categories, name="right"),
    )
