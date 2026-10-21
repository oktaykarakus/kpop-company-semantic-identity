from __future__ import annotations

import numpy as np
import pandas as pd


def descriptive_statistics(
    df: pd.DataFrame,
    value: str,
    group: str,
) -> pd.DataFrame:
    """
    Calculate descriptive statistics for one variable by group.
    """

    rows = []

    for name, subset in df.groupby(group, observed=True):

        x = subset[value].dropna().to_numpy()

        if len(x) == 0:
            continue

        q1 = np.percentile(x, 25)
        q3 = np.percentile(x, 75)

        se = np.std(x, ddof=1) / np.sqrt(len(x))
        ci = 1.96 * se

        rows.append(
            {
                group: name,
                "n": len(x),
                "mean": np.mean(x),
                "median": np.median(x),
                "std": np.std(x, ddof=1),
                "min": np.min(x),
                "max": np.max(x),
                "q1": q1,
                "q3": q3,
                "iqr": q3 - q1,
                "ci_low": np.mean(x) - ci,
                "ci_high": np.mean(x) + ci,
            }
        )

    return pd.DataFrame(rows)