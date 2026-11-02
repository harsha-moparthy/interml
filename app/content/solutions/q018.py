import pandas as pd


def q018(frames):
    """Stack frames with a fresh row index; return an empty frame for an empty list."""
    return (
        pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    )
