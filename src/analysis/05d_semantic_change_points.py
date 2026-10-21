#!/usr/bin/env python3
"""
=========================================================
WP5 - Temporal Dynamics
05d_semantic_change_points.py

Scientific question
-------------------
When did entertainment companies undergo meaningful
semantic shifts?

This script detects unusually large company-level semantic
changes using a robust, company-specific threshold based on
the median and median absolute deviation (MAD).

It then groups adjacent transition-level detections into
broader semantic change episodes. This prevents consecutive
detections from heavily overlapping rolling windows being
interpreted as fully independent historical events.

Method
------
For each company:

    median_js = median(JS divergence)

    MAD = median(
        absolute(JS divergence - median_js)
    )

    robust_sigma = 1.4826 * MAD

    threshold = median_js + 3 * robust_sigma

A transition is classified as a semantic change point when
its Jensen-Shannon divergence exceeds the robust threshold.

Only consecutive one-year rolling-window transitions are
used for formal change-point detection.

Severity
--------
Transitions are classified using robust z-scores:

    ordinary:
        robust z < 2

    elevated:
        2 <= robust z < 3

    change_point:
        3 <= robust z < 5

    major_change_point:
        robust z >= 5

Semantic change episodes
------------------------
Adjacent detected transitions are grouped into the same
episode when their rolling-window start years are consecutive.

For example:

    2003-2007 -> 2004-2008

followed by:

    2004-2008 -> 2005-2009

forms one semantic change episode because the two detections
originate from consecutive rolling-window transitions and
share four of five years.

An episode ends when the next detected transition is not
temporally adjacent.

Episode outputs include:

    - Episode start and end years
    - Number of transition-level detections
    - Episode duration
    - Peak JS divergence
    - Mean JS divergence
    - Median JS divergence
    - Cumulative JS divergence
    - Maximum robust z-score
    - Peak transition
    - Episode severity
    - All transitions in the episode

Input
-----
outputs/temporal/rolling5/
    semantic_drift.parquet

Outputs
-------
outputs/temporal/rolling5/
    semantic_change_points.csv
    semantic_change_point_summary.csv
    semantic_change_point_method.csv
    semantic_change_episodes.csv
    semantic_change_episode_summary.csv
=========================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------
# Configuration
# --------------------------------------------------------

MAD_CONSISTENCY_CONSTANT = 1.4826

CHANGE_POINT_Z_THRESHOLD = 3.0

MAJOR_CHANGE_POINT_Z_THRESHOLD = 5.0

ELEVATED_Z_THRESHOLD = 2.0

IQR_TO_SIGMA_CONSTANT = 1.349

EPSILON = 1e-12


# --------------------------------------------------------
# Paths
# --------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "outputs"
    / "temporal"
    / "rolling5"
    / "semantic_drift.parquet"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "temporal"
    / "rolling5"
)

CHANGE_POINTS_FILE = (
    OUTPUT_DIR
    / "semantic_change_points.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "semantic_change_point_summary.csv"
)

METHOD_FILE = (
    OUTPUT_DIR
    / "semantic_change_point_method.csv"
)

EPISODES_FILE = (
    OUTPUT_DIR
    / "semantic_change_episodes.csv"
)

EPISODE_SUMMARY_FILE = (
    OUTPUT_DIR
    / "semantic_change_episode_summary.csv"
)


# --------------------------------------------------------
# Validation
# --------------------------------------------------------

def validate_input(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate the semantic-drift dataframe.
    """

    required_columns = {
        "company",
        "from_window",
        "to_window",
        "from_start",
        "to_start",
        "year_gap",
        "is_consecutive",
        "js_divergence",
    }

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        missing_text = ", ".join(
            sorted(missing_columns)
        )

        raise ValueError(
            "Input file is missing required columns: "
            f"{missing_text}"
        )

    if dataframe.empty:
        raise ValueError(
            "The semantic-drift dataframe is empty."
        )

    if dataframe["company"].isna().any():
        raise ValueError(
            "The input dataframe contains missing company "
            "names."
        )

    if dataframe["js_divergence"].isna().any():
        raise ValueError(
            "The input dataframe contains missing "
            "Jensen-Shannon divergence values."
        )

    consecutive_mask = (
        dataframe["is_consecutive"]
        .astype(bool)
    )

    if not consecutive_mask.any():
        raise ValueError(
            "No consecutive transitions are available for "
            "change-point detection."
        )


def validate_company_data(
    company_df: pd.DataFrame,
    company: str,
) -> None:
    """
    Validate one company's consecutive transition series.
    """

    if company_df.empty:
        raise ValueError(
            f"No valid transitions found for '{company}'."
        )

    duplicate_transitions = (
        company_df[
            [
                "from_start",
                "to_start",
            ]
        ]
        .duplicated(
            keep=False
        )
    )

    if duplicate_transitions.any():
        duplicated_rows = (
            company_df.loc[
                duplicate_transitions,
                [
                    "from_start",
                    "to_start",
                ],
            ]
            .astype(str)
            .agg("->".join, axis=1)
            .tolist()
        )

        raise ValueError(
            "Duplicate transitions detected for company "
            f"'{company}': {duplicated_rows}"
        )

    if not (
        company_df["year_gap"] == 1
    ).all():
        raise ValueError(
            "Formal change-point detection requires "
            f"year_gap == 1 for company '{company}'."
        )


# --------------------------------------------------------
# Robust scale estimation
# --------------------------------------------------------

def calculate_robust_scale(
    values: pd.Series,
) -> dict:
    """
    Calculate robust location and scale statistics.
    """

    numeric_values = (
        pd.to_numeric(
            values,
            errors="coerce",
        )
        .dropna()
        .astype(float)
    )

    if numeric_values.empty:
        raise ValueError(
            "Cannot estimate robust scale from an empty "
            "series."
        )

    median_js = float(
        numeric_values.median()
    )

    absolute_deviations = (
        numeric_values
        - median_js
    ).abs()

    mad = float(
        absolute_deviations.median()
    )

    q1 = float(
        numeric_values.quantile(
            0.25
        )
    )

    q3 = float(
        numeric_values.quantile(
            0.75
        )
    )

    iqr = q3 - q1

    mad_scale = (
        MAD_CONSISTENCY_CONSTANT
        * mad
    )

    if mad_scale > EPSILON:

        robust_sigma = mad_scale

        scale_method = (
            "consistency_adjusted_mad"
        )

    elif iqr > EPSILON:

        robust_sigma = (
            iqr
            / IQR_TO_SIGMA_CONSTANT
        )

        scale_method = "iqr_fallback"

    else:

        robust_sigma = 0.0

        scale_method = "zero_variation"

    threshold = (
        median_js
        + (
            CHANGE_POINT_Z_THRESHOLD
            * robust_sigma
        )
    )

    major_threshold = (
        median_js
        + (
            MAJOR_CHANGE_POINT_Z_THRESHOLD
            * robust_sigma
        )
    )

    return {
        "median_js": median_js,
        "mad": mad,
        "q1_js": q1,
        "q3_js": q3,
        "iqr_js": iqr,
        "robust_sigma": robust_sigma,
        "scale_method": scale_method,
        "threshold": threshold,
        "major_threshold": major_threshold,
    }


def calculate_robust_z_scores(
    values: pd.Series,
    median_js: float,
    robust_sigma: float,
) -> pd.Series:
    """
    Calculate upper-tail robust z-scores.
    """

    numeric_values = pd.to_numeric(
        values,
        errors="coerce",
    ).astype(float)

    if robust_sigma > EPSILON:

        return (
            numeric_values
            - median_js
        ) / robust_sigma

    robust_z = pd.Series(
        np.nan,
        index=numeric_values.index,
        dtype=float,
    )

    equal_mask = np.isclose(
        numeric_values,
        median_js,
        atol=EPSILON,
        rtol=0.0,
    )

    above_mask = (
        numeric_values
        > median_js
        + EPSILON
    )

    below_mask = (
        numeric_values
        < median_js
        - EPSILON
    )

    robust_z.loc[
        equal_mask
    ] = 0.0

    robust_z.loc[
        above_mask
    ] = np.inf

    robust_z.loc[
        below_mask
    ] = -np.inf

    return robust_z


# --------------------------------------------------------
# Transition classification
# --------------------------------------------------------

def classify_severity(
    robust_z: float,
    change_point: bool,
) -> str:
    """
    Assign an interpretable transition severity category.
    """

    if pd.isna(
        robust_z
    ):
        return "undefined"

    if (
        change_point
        and robust_z
        >= MAJOR_CHANGE_POINT_Z_THRESHOLD
    ):
        return "major_change_point"

    if change_point:
        return "change_point"

    if robust_z >= ELEVATED_Z_THRESHOLD:
        return "elevated"

    return "ordinary"


def calculate_exceedance_ratio(
    js_divergence: float,
    threshold: float,
) -> float:
    """
    Calculate drift magnitude relative to the threshold.
    """

    if threshold > EPSILON:

        return float(
            js_divergence
            / threshold
        )

    if js_divergence > threshold:

        return np.inf

    return 1.0


# --------------------------------------------------------
# Company-level change-point detection
# --------------------------------------------------------

def detect_company_change_points(
    company_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Detect semantic change points for one company.
    """

    company_df = (
        company_df
        .sort_values(
            [
                "from_start",
                "to_start",
            ]
        )
        .reset_index(drop=True)
        .copy()
    )

    company = str(
        company_df["company"].iloc[0]
    )

    validate_company_data(
        company_df=company_df,
        company=company,
    )

    scale_statistics = (
        calculate_robust_scale(
            company_df["js_divergence"]
        )
    )

    company_df["transition"] = (
        company_df["from_window"]
        .astype(str)
        + " -> "
        + company_df["to_window"]
        .astype(str)
    )

    company_df["median_js"] = (
        scale_statistics["median_js"]
    )

    company_df["mad"] = (
        scale_statistics["mad"]
    )

    company_df["q1_js"] = (
        scale_statistics["q1_js"]
    )

    company_df["q3_js"] = (
        scale_statistics["q3_js"]
    )

    company_df["iqr_js"] = (
        scale_statistics["iqr_js"]
    )

    company_df["robust_sigma"] = (
        scale_statistics["robust_sigma"]
    )

    company_df["scale_method"] = (
        scale_statistics["scale_method"]
    )

    company_df["threshold"] = (
        scale_statistics["threshold"]
    )

    company_df["major_threshold"] = (
        scale_statistics["major_threshold"]
    )

    company_df["robust_z"] = (
        calculate_robust_z_scores(
            values=company_df[
                "js_divergence"
            ],
            median_js=scale_statistics[
                "median_js"
            ],
            robust_sigma=scale_statistics[
                "robust_sigma"
            ],
        )
    )

    company_df["change_point"] = (
        company_df["robust_z"]
        .ge(
            CHANGE_POINT_Z_THRESHOLD
        )
        .fillna(False)
    )

    company_df["severity"] = [
        classify_severity(
            robust_z=robust_z,
            change_point=change_point,
        )
        for robust_z, change_point
        in zip(
            company_df["robust_z"],
            company_df["change_point"],
        )
    ]

    company_df["excess_above_threshold"] = (
        company_df["js_divergence"]
        - company_df["threshold"]
    )

    company_df["threshold_ratio"] = [
        calculate_exceedance_ratio(
            js_divergence=float(
                js_divergence
            ),
            threshold=float(
                threshold
            ),
        )
        for js_divergence, threshold
        in zip(
            company_df["js_divergence"],
            company_df["threshold"],
        )
    ]

    company_df["company_change_rank"] = (
        company_df["js_divergence"]
        .rank(
            method="dense",
            ascending=False,
        )
        .astype(int)
    )

    method_row = {
        "company": company,
        "n_transitions": int(
            len(company_df)
        ),
        **scale_statistics,
        "change_point_z_threshold": (
            CHANGE_POINT_Z_THRESHOLD
        ),
        "major_change_point_z_threshold": (
            MAJOR_CHANGE_POINT_Z_THRESHOLD
        ),
        "n_change_points": int(
            company_df["change_point"].sum()
        ),
    }

    return (
        company_df,
        method_row,
    )


# --------------------------------------------------------
# Change-point summary
# --------------------------------------------------------

def build_company_summary(
    change_points_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a concise company-level change-point summary.
    """

    summary_rows = []

    for company, company_df in (
        change_points_df.groupby(
            "company",
            sort=True,
        )
    ):

        company_df = (
            company_df
            .sort_values(
                [
                    "from_start",
                    "to_start",
                ]
            )
            .reset_index(drop=True)
        )

        detected = (
            company_df.loc[
                company_df["change_point"]
            ]
            .sort_values(
                [
                    "js_divergence",
                    "from_start",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .reset_index(drop=True)
        )

        largest_index = (
            company_df[
                "js_divergence"
            ].idxmax()
        )

        largest_row = (
            company_df.loc[
                largest_index
            ]
        )

        if detected.empty:

            first_change_transition = ""

            first_change_from_year = np.nan

            first_change_to_year = np.nan

            largest_detected_transition = ""

            largest_detected_js = np.nan

            largest_detected_severity = ""

            change_years = ""

            change_transitions = ""

        else:

            chronological_detected = (
                detected
                .sort_values(
                    [
                        "from_start",
                        "to_start",
                    ]
                )
                .reset_index(drop=True)
            )

            first_change_row = (
                chronological_detected.iloc[0]
            )

            largest_detected_row = (
                detected.iloc[0]
            )

            first_change_transition = str(
                first_change_row[
                    "transition"
                ]
            )

            first_change_from_year = int(
                first_change_row[
                    "from_start"
                ]
            )

            first_change_to_year = int(
                first_change_row[
                    "to_start"
                ]
            )

            largest_detected_transition = str(
                largest_detected_row[
                    "transition"
                ]
            )

            largest_detected_js = float(
                largest_detected_row[
                    "js_divergence"
                ]
            )

            largest_detected_severity = str(
                largest_detected_row[
                    "severity"
                ]
            )

            change_years = "|".join(
                chronological_detected[
                    "to_start"
                ]
                .astype(int)
                .astype(str)
                .tolist()
            )

            change_transitions = "|".join(
                chronological_detected[
                    "transition"
                ]
                .astype(str)
                .tolist()
            )

        summary_rows.append(
            {
                "company": company,
                "n_transitions": int(
                    len(company_df)
                ),
                "n_change_points": int(
                    company_df[
                        "change_point"
                    ].sum()
                ),
                "change_point_rate": float(
                    company_df[
                        "change_point"
                    ].mean()
                ),
                "median_js": float(
                    company_df[
                        "median_js"
                    ].iloc[0]
                ),
                "mad": float(
                    company_df[
                        "mad"
                    ].iloc[0]
                ),
                "robust_sigma": float(
                    company_df[
                        "robust_sigma"
                    ].iloc[0]
                ),
                "threshold": float(
                    company_df[
                        "threshold"
                    ].iloc[0]
                ),
                "scale_method": str(
                    company_df[
                        "scale_method"
                    ].iloc[0]
                ),
                "largest_overall_transition": str(
                    largest_row[
                        "transition"
                    ]
                ),
                "largest_overall_js": float(
                    largest_row[
                        "js_divergence"
                    ]
                ),
                "largest_overall_is_change_point": bool(
                    largest_row[
                        "change_point"
                    ]
                ),
                "largest_detected_transition": (
                    largest_detected_transition
                ),
                "largest_detected_js": (
                    largest_detected_js
                ),
                "largest_detected_severity": (
                    largest_detected_severity
                ),
                "first_change_transition": (
                    first_change_transition
                ),
                "first_change_from_year": (
                    first_change_from_year
                ),
                "first_change_to_year": (
                    first_change_to_year
                ),
                "change_years": change_years,
                "change_transitions": (
                    change_transitions
                ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    return (
        summary
        .sort_values(
            [
                "n_change_points",
                "largest_detected_js",
                "largest_overall_js",
            ],
            ascending=[
                False,
                False,
                False,
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )


# --------------------------------------------------------
# Semantic change episodes
# --------------------------------------------------------

def classify_episode_severity(
    peak_robust_z: float,
    n_change_points: int,
) -> str:
    """
    Classify a semantic change episode.

    Episode severity reflects both peak statistical
    extremeness and whether the episode persists across
    multiple overlapping transitions.
    """

    if pd.isna(
        peak_robust_z
    ):
        return "undefined"

    if (
        peak_robust_z
        >= MAJOR_CHANGE_POINT_Z_THRESHOLD
        and n_change_points >= 2
    ):
        return "sustained_major_episode"

    if (
        peak_robust_z
        >= MAJOR_CHANGE_POINT_Z_THRESHOLD
    ):
        return "major_episode"

    if n_change_points >= 2:
        return "sustained_change_episode"

    return "change_episode"


def build_episode_row(
    episode_df: pd.DataFrame,
    episode_number: int,
) -> dict:
    """
    Summarise one group of adjacent change-point detections.
    """

    episode_df = (
        episode_df
        .sort_values(
            [
                "from_start",
                "to_start",
            ]
        )
        .reset_index(drop=True)
    )

    company = str(
        episode_df["company"].iloc[0]
    )

    peak_index = (
        episode_df[
            "js_divergence"
        ].idxmax()
    )

    peak_row = (
        episode_df.loc[
            peak_index
        ]
    )

    first_row = (
        episode_df.iloc[0]
    )

    last_row = (
        episode_df.iloc[-1]
    )

    n_change_points = int(
        len(episode_df)
    )

    episode_start_year = int(
        first_row["from_start"]
    )

    episode_end_year = int(
        last_row["to_start"]
    )

    episode_duration_years = int(
        episode_end_year
        - episode_start_year
        + 1
    )

    peak_robust_z = float(
        episode_df[
            "robust_z"
        ].max()
    )

    episode_severity = (
        classify_episode_severity(
            peak_robust_z=peak_robust_z,
            n_change_points=n_change_points,
        )
    )

    transition_list = "|".join(
        episode_df[
            "transition"
        ]
        .astype(str)
        .tolist()
    )

    transition_years = "|".join(
        episode_df[
            "to_start"
        ]
        .astype(int)
        .astype(str)
        .tolist()
    )

    severity_sequence = "|".join(
        episode_df[
            "severity"
        ]
        .astype(str)
        .tolist()
    )

    return {
        "company": company,
        "episode_number": int(
            episode_number
        ),
        "episode_label": (
            f"{company}_episode_{episode_number}"
        ),
        "episode_start_year": (
            episode_start_year
        ),
        "episode_end_year": (
            episode_end_year
        ),
        "episode_duration_years": (
            episode_duration_years
        ),
        "start_from_window": str(
            first_row["from_window"]
        ),
        "end_to_window": str(
            last_row["to_window"]
        ),
        "n_change_points": (
            n_change_points
        ),
        "peak_js_divergence": float(
            peak_row[
                "js_divergence"
            ]
        ),
        "mean_js_divergence": float(
            episode_df[
                "js_divergence"
            ].mean()
        ),
        "median_js_divergence": float(
            episode_df[
                "js_divergence"
            ].median()
        ),
        "cumulative_js_divergence": float(
            episode_df[
                "js_divergence"
            ].sum()
        ),
        "minimum_js_divergence": float(
            episode_df[
                "js_divergence"
            ].min()
        ),
        "maximum_robust_z": (
            peak_robust_z
        ),
        "mean_robust_z": float(
            episode_df[
                "robust_z"
            ].mean()
        ),
        "peak_transition": str(
            peak_row[
                "transition"
            ]
        ),
        "peak_transition_from_year": int(
            peak_row[
                "from_start"
            ]
        ),
        "peak_transition_to_year": int(
            peak_row[
                "to_start"
            ]
        ),
        "episode_severity": (
            episode_severity
        ),
        "transition_years": (
            transition_years
        ),
        "transitions": transition_list,
        "severity_sequence": (
            severity_sequence
        ),
    }


def detect_company_change_episodes(
    company_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Group adjacent change-point detections for one company.

    Two detections belong to the same episode when the
    from_start year of the next detected transition is one
    year after that of the previous detected transition.
    """

    detected = (
        company_df.loc[
            company_df["change_point"]
        ]
        .sort_values(
            [
                "from_start",
                "to_start",
            ]
        )
        .reset_index(drop=True)
        .copy()
    )

    episode_columns = [
        "company",
        "episode_number",
        "episode_label",
        "episode_start_year",
        "episode_end_year",
        "episode_duration_years",
        "start_from_window",
        "end_to_window",
        "n_change_points",
        "peak_js_divergence",
        "mean_js_divergence",
        "median_js_divergence",
        "cumulative_js_divergence",
        "minimum_js_divergence",
        "maximum_robust_z",
        "mean_robust_z",
        "peak_transition",
        "peak_transition_from_year",
        "peak_transition_to_year",
        "episode_severity",
        "transition_years",
        "transitions",
        "severity_sequence",
    ]

    if detected.empty:
        return pd.DataFrame(
            columns=episode_columns
        )

    episode_rows = []

    current_indices = [
        0
    ]

    episode_number = 1

    for row_index in range(
        1,
        len(detected),
    ):

        previous_row = (
            detected.iloc[
                row_index - 1
            ]
        )

        current_row = (
            detected.iloc[
                row_index
            ]
        )

        is_adjacent = (
            int(
                current_row[
                    "from_start"
                ]
            )
            == int(
                previous_row[
                    "from_start"
                ]
            )
            + 1
        )

        if is_adjacent:

            current_indices.append(
                row_index
            )

        else:

            episode_df = (
                detected.iloc[
                    current_indices
                ]
                .copy()
            )

            episode_rows.append(
                build_episode_row(
                    episode_df=episode_df,
                    episode_number=episode_number,
                )
            )

            episode_number += 1

            current_indices = [
                row_index
            ]

    final_episode_df = (
        detected.iloc[
            current_indices
        ]
        .copy()
    )

    episode_rows.append(
        build_episode_row(
            episode_df=final_episode_df,
            episode_number=episode_number,
        )
    )

    return pd.DataFrame(
        episode_rows,
        columns=episode_columns,
    )


def detect_all_change_episodes(
    change_points_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Detect semantic change episodes for all companies.
    """

    episode_frames = []

    for company, company_df in (
        change_points_df.groupby(
            "company",
            sort=True,
        )
    ):

        company_episodes = (
            detect_company_change_episodes(
                company_df
            )
        )

        if not company_episodes.empty:

            episode_frames.append(
                company_episodes
            )

    if not episode_frames:

        return pd.DataFrame(
            columns=[
                "company",
                "episode_number",
                "episode_label",
                "episode_start_year",
                "episode_end_year",
                "episode_duration_years",
                "start_from_window",
                "end_to_window",
                "n_change_points",
                "peak_js_divergence",
                "mean_js_divergence",
                "median_js_divergence",
                "cumulative_js_divergence",
                "minimum_js_divergence",
                "maximum_robust_z",
                "mean_robust_z",
                "peak_transition",
                "peak_transition_from_year",
                "peak_transition_to_year",
                "episode_severity",
                "transition_years",
                "transitions",
                "severity_sequence",
            ]
        )

    episodes = pd.concat(
        episode_frames,
        ignore_index=True,
    )

    return (
        episodes
        .sort_values(
            [
                "company",
                "episode_start_year",
                "episode_number",
            ]
        )
        .reset_index(drop=True)
    )


def build_episode_summary(
    episodes: pd.DataFrame,
    change_points: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one episode-level summary row per company.
    """

    companies = sorted(
        change_points[
            "company"
        ].unique()
    )

    summary_rows = []

    for company in companies:

        company_episodes = (
            episodes.loc[
                episodes[
                    "company"
                ] == company
            ]
            .copy()
        )

        company_change_points = (
            change_points.loc[
                (
                    change_points[
                        "company"
                    ] == company
                )
                & (
                    change_points[
                        "change_point"
                    ]
                )
            ]
            .copy()
        )

        n_transition_detections = int(
            len(
                company_change_points
            )
        )

        if company_episodes.empty:

            n_episodes = 0

            largest_episode_number = np.nan

            largest_episode_start_year = np.nan

            largest_episode_end_year = np.nan

            largest_episode_peak_js = np.nan

            largest_episode_cumulative_js = np.nan

            largest_episode_severity = ""

            largest_episode_peak_transition = ""

            longest_episode_duration = np.nan

            episode_periods = ""

        else:

            largest_episode = (
                company_episodes
                .sort_values(
                    [
                        "peak_js_divergence",
                        "cumulative_js_divergence",
                        "episode_start_year",
                    ],
                    ascending=[
                        False,
                        False,
                        True,
                    ],
                )
                .iloc[0]
            )

            longest_episode = (
                company_episodes
                .sort_values(
                    [
                        "episode_duration_years",
                        "n_change_points",
                        "peak_js_divergence",
                    ],
                    ascending=[
                        False,
                        False,
                        False,
                    ],
                )
                .iloc[0]
            )

            n_episodes = int(
                len(company_episodes)
            )

            largest_episode_number = int(
                largest_episode[
                    "episode_number"
                ]
            )

            largest_episode_start_year = int(
                largest_episode[
                    "episode_start_year"
                ]
            )

            largest_episode_end_year = int(
                largest_episode[
                    "episode_end_year"
                ]
            )

            largest_episode_peak_js = float(
                largest_episode[
                    "peak_js_divergence"
                ]
            )

            largest_episode_cumulative_js = float(
                largest_episode[
                    "cumulative_js_divergence"
                ]
            )

            largest_episode_severity = str(
                largest_episode[
                    "episode_severity"
                ]
            )

            largest_episode_peak_transition = str(
                largest_episode[
                    "peak_transition"
                ]
            )

            longest_episode_duration = int(
                longest_episode[
                    "episode_duration_years"
                ]
            )

            episode_periods = "|".join(
                (
                    company_episodes[
                        "episode_start_year"
                    ]
                    .astype(int)
                    .astype(str)
                    + "-"
                    + company_episodes[
                        "episode_end_year"
                    ]
                    .astype(int)
                    .astype(str)
                )
                .tolist()
            )

        compression_ratio = (
            float(
                n_episodes
                / n_transition_detections
            )
            if n_transition_detections > 0
            else np.nan
        )

        summary_rows.append(
            {
                "company": company,
                "n_transition_level_detections": (
                    n_transition_detections
                ),
                "n_change_episodes": (
                    n_episodes
                ),
                "episode_compression_ratio": (
                    compression_ratio
                ),
                "largest_episode_number": (
                    largest_episode_number
                ),
                "largest_episode_start_year": (
                    largest_episode_start_year
                ),
                "largest_episode_end_year": (
                    largest_episode_end_year
                ),
                "largest_episode_peak_js": (
                    largest_episode_peak_js
                ),
                "largest_episode_cumulative_js": (
                    largest_episode_cumulative_js
                ),
                "largest_episode_severity": (
                    largest_episode_severity
                ),
                "largest_episode_peak_transition": (
                    largest_episode_peak_transition
                ),
                "longest_episode_duration_years": (
                    longest_episode_duration
                ),
                "episode_periods": (
                    episode_periods
                ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    return (
        summary
        .sort_values(
            [
                "n_change_episodes",
                "largest_episode_peak_js",
            ],
            ascending=[
                False,
                False,
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )


# --------------------------------------------------------
# Main
# --------------------------------------------------------

def main() -> None:
    """
    Run robust semantic change-point and episode detection.
    """

    print("=" * 70)
    print("WP5 : SEMANTIC CHANGE-POINT DETECTION")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------
    # Load semantic drift
    # ----------------------------------------------------

    drift = pd.read_parquet(
        INPUT_FILE
    )

    validate_input(
        drift
    )

    drift = drift.copy()

    drift["company"] = (
        drift["company"]
        .astype(str)
    )

    drift["from_window"] = (
        drift["from_window"]
        .astype(str)
    )

    drift["to_window"] = (
        drift["to_window"]
        .astype(str)
    )

    integer_columns = [
        "from_start",
        "to_start",
        "year_gap",
    ]

    for column in integer_columns:

        drift[column] = pd.to_numeric(
            drift[column],
            errors="raise",
        ).astype(int)

    drift["js_divergence"] = pd.to_numeric(
        drift["js_divergence"],
        errors="raise",
    ).astype(float)

    consecutive_drift = (
        drift.loc[
            drift["is_consecutive"]
            .astype(bool)
        ]
        .copy()
        .reset_index(drop=True)
    )

    irregular_count = int(
        (
            ~drift["is_consecutive"]
            .astype(bool)
        ).sum()
    )

    companies = sorted(
        consecutive_drift[
            "company"
        ].unique()
    )

    print()
    print(f"Input file             : {INPUT_FILE}")
    print(f"All transitions loaded : {len(drift)}")
    print(
        f"Consecutive transitions: "
        f"{len(consecutive_drift)}"
    )
    print(
        f"Irregular transitions  : "
        f"{irregular_count}"
    )
    print(f"Companies found        : {len(companies)}")
    print(
        "Change-point threshold : "
        f"median + {CHANGE_POINT_Z_THRESHOLD} "
        "robust sigma"
    )

    # ----------------------------------------------------
    # Detect transition-level change points
    # ----------------------------------------------------

    company_results = []

    method_rows = []

    for company in companies:

        company_df = (
            consecutive_drift.loc[
                consecutive_drift[
                    "company"
                ] == company
            ]
            .copy()
        )

        (
            company_change_points,
            method_row,
        ) = detect_company_change_points(
            company_df
        )

        company_results.append(
            company_change_points
        )

        method_rows.append(
            method_row
        )

    change_points = pd.concat(
        company_results,
        ignore_index=True,
    )

    change_points = (
        change_points
        .sort_values(
            [
                "company",
                "from_start",
                "to_start",
            ]
        )
        .reset_index(drop=True)
    )

    method_summary = pd.DataFrame(
        method_rows
    )

    method_summary = (
        method_summary
        .sort_values(
            "company"
        )
        .reset_index(drop=True)
    )

    company_summary = build_company_summary(
        change_points
    )

    # ----------------------------------------------------
    # Detect semantic change episodes
    # ----------------------------------------------------

    episodes = detect_all_change_episodes(
        change_points
    )

    episode_summary = build_episode_summary(
        episodes=episodes,
        change_points=change_points,
    )

    # ----------------------------------------------------
    # Select transition-level output columns
    # ----------------------------------------------------

    change_point_columns = [
        "company",
        "transition",
        "from_window",
        "to_window",
        "from_start",
        "to_start",
        "year_gap",
        "js_divergence",
        "median_js",
        "mad",
        "q1_js",
        "q3_js",
        "iqr_js",
        "robust_sigma",
        "scale_method",
        "threshold",
        "major_threshold",
        "robust_z",
        "excess_above_threshold",
        "threshold_ratio",
        "change_point",
        "severity",
        "company_change_rank",
    ]

    change_points_output = (
        change_points[
            change_point_columns
        ]
        .copy()
    )

    # ----------------------------------------------------
    # Save outputs
    # ----------------------------------------------------

    change_points_output.to_csv(
        CHANGE_POINTS_FILE,
        index=False,
    )

    company_summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    method_summary.to_csv(
        METHOD_FILE,
        index=False,
    )

    episodes.to_csv(
        EPISODES_FILE,
        index=False,
    )

    episode_summary.to_csv(
        EPISODE_SUMMARY_FILE,
        index=False,
    )

    # ----------------------------------------------------
    # Console report
    # ----------------------------------------------------

    detected_change_points = (
        change_points_output.loc[
            change_points_output[
                "change_point"
            ]
        ]
        .copy()
    )

    print()
    print("=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)

    print(
        f"Companies analysed          : "
        f"{len(companies)}"
    )

    print(
        f"Transitions analysed        : "
        f"{len(change_points_output)}"
    )

    print(
        f"Transition-level detections : "
        f"{len(detected_change_points)}"
    )

    print(
        f"Semantic change episodes    : "
        f"{len(episodes)}"
    )

    print()
    print("=" * 70)
    print("COMPANY CHANGE-POINT SUMMARY")
    print("=" * 70)

    summary_display_columns = [
        "company",
        "n_transitions",
        "n_change_points",
        "change_point_rate",
        "median_js",
        "mad",
        "threshold",
        "largest_overall_transition",
        "largest_overall_js",
        "largest_detected_transition",
        "largest_detected_js",
        "change_years",
    ]

    print(
        company_summary[
            summary_display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("DETECTED SEMANTIC CHANGE POINTS")
    print("=" * 70)

    if detected_change_points.empty:

        print(
            "No semantic change points were detected."
        )

    else:

        detected_display_columns = [
            "company",
            "transition",
            "js_divergence",
            "median_js",
            "mad",
            "threshold",
            "robust_z",
            "severity",
        ]

        detected_change_points = (
            detected_change_points
            .sort_values(
                [
                    "company",
                    "from_start",
                    "to_start",
                ]
            )
        )

        print(
            detected_change_points[
                detected_display_columns
            ].to_string(
                index=False
            )
        )

    print()
    print("=" * 70)
    print("SEMANTIC CHANGE EPISODES")
    print("=" * 70)

    if episodes.empty:

        print(
            "No semantic change episodes were detected."
        )

    else:

        episode_display_columns = [
            "company",
            "episode_number",
            "episode_start_year",
            "episode_end_year",
            "episode_duration_years",
            "n_change_points",
            "peak_js_divergence",
            "mean_js_divergence",
            "cumulative_js_divergence",
            "maximum_robust_z",
            "episode_severity",
            "peak_transition",
        ]

        print(
            episodes[
                episode_display_columns
            ].to_string(
                index=False
            )
        )

    print()
    print("=" * 70)
    print("COMPANY EPISODE SUMMARY")
    print("=" * 70)

    episode_summary_display_columns = [
        "company",
        "n_transition_level_detections",
        "n_change_episodes",
        "episode_compression_ratio",
        "largest_episode_start_year",
        "largest_episode_end_year",
        "largest_episode_peak_js",
        "largest_episode_severity",
        "longest_episode_duration_years",
        "episode_periods",
    ]

    print(
        episode_summary[
            episode_summary_display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("ROBUST THRESHOLDS")
    print("=" * 70)

    method_display_columns = [
        "company",
        "n_transitions",
        "median_js",
        "mad",
        "iqr_js",
        "robust_sigma",
        "scale_method",
        "threshold",
        "major_threshold",
        "n_change_points",
    ]

    print(
        method_summary[
            method_display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("FILES WRITTEN")
    print("=" * 70)

    print(CHANGE_POINTS_FILE)
    print(SUMMARY_FILE)
    print(METHOD_FILE)
    print(EPISODES_FILE)
    print(EPISODE_SUMMARY_FILE)

    print("=" * 70)


if __name__ == "__main__":
    main()