#!/usr/bin/env python3
"""
=========================================================
WP6 - Robustness and Sensitivity Analysis
06a_similarity_metric_sensitivity.py

Scientific question
-------------------
Do the principal semantic-fingerprint conclusions remain
consistent when alternative distance metrics are used?

Purpose
-------
The principal company-comparison analyses use cosine-based
comparison of semantic fingerprints.

This script evaluates whether the conclusions remain stable
when the same company fingerprints are compared using:

    - Cosine distance
    - Euclidean distance
    - Correlation distance

The semantic fingerprints themselves are not redesigned.

Static company fingerprints are reconstructed directly from
the WP3/WP4 master dataset as company-level probability
distributions over the existing semantic communities.

Temporal fingerprints are read directly from the existing
rolling five-year WP5 output.

Analyses
--------
1. Static pairwise company distances

   For each metric, calculate all pairwise distances between
   company semantic fingerprints.

2. Static company distinctiveness rankings

   Rank companies by their mean distance from all other
   companies.

3. Pairwise fingerprint agreement

   Compare the upper triangles of the three company-distance
   matrices using:

       - Spearman correlation
       - Kendall tau
       - Pearson correlation
       - Mean absolute difference after min-max scaling

4. Temporal drift sensitivity

   Recalculate consecutive rolling-five-year drift using
   cosine, Euclidean, and correlation distance.

   Rank companies by mean temporal drift under each metric.

5. Ranking agreement

   Compare static and temporal rankings using:

       - Spearman correlation
       - Kendall tau
       - Exact-rank agreement
       - Mean absolute rank difference
       - Most-distinctive-company agreement
       - Most-dynamic-company agreement
       - Most-stable-company agreement

Important interpretation
------------------------
This script changes only the distance metric.

It does not:

    - change the community partition,
    - rerun graph construction,
    - redefine company fingerprints,
    - modify temporal windows,
    - or introduce new semantic features.

Inputs
------
outputs/tables/
    company_master_analysis_v1.parquet

outputs/temporal/rolling5/
    community_profiles.parquet

Outputs
-------
outputs/robustness/metric_sensitivity/

    metric_sensitivity_company_profiles.csv
    metric_sensitivity_company_pairs.csv
    metric_sensitivity_company_rankings.csv
    metric_sensitivity_temporal_transitions.csv
    metric_sensitivity_temporal_rankings.csv
    metric_sensitivity_agreement.csv
    metric_sensitivity_summary.csv
=========================================================
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.spatial.distance import (
    correlation,
    cosine,
    euclidean,
)

from scipy.stats import (
    kendalltau,
    pearsonr,
    spearmanr,
)


# --------------------------------------------------------
# Configuration
# --------------------------------------------------------

METRICS = [
    "cosine",
    "euclidean",
    "correlation",
]

BASELINE_METRIC = "cosine"

EPSILON = 1e-12


# --------------------------------------------------------
# Paths
# --------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

MASTER_DATA_FILE = (
    ROOT
    / "outputs"
    / "tables"
    / "company_master_analysis_v1.parquet"
)

TEMPORAL_PROFILE_FILE = (
    ROOT
    / "outputs"
    / "temporal"
    / "rolling5"
    / "community_profiles.parquet"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "robustness"
    / "metric_sensitivity"
)

COMPANY_PROFILES_FILE = (
    OUTPUT_DIR
    / "metric_sensitivity_company_profiles.csv"
)

COMPANY_PAIRS_FILE = (
    OUTPUT_DIR
    / "metric_sensitivity_company_pairs.csv"
)

COMPANY_RANKINGS_FILE = (
    OUTPUT_DIR
    / "metric_sensitivity_company_rankings.csv"
)

TEMPORAL_TRANSITIONS_FILE = (
    OUTPUT_DIR
    / "metric_sensitivity_temporal_transitions.csv"
)

TEMPORAL_RANKINGS_FILE = (
    OUTPUT_DIR
    / "metric_sensitivity_temporal_rankings.csv"
)

AGREEMENT_FILE = (
    OUTPUT_DIR
    / "metric_sensitivity_agreement.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "metric_sensitivity_summary.csv"
)


# --------------------------------------------------------
# Input validation
# --------------------------------------------------------

def validate_master_data(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate the WP3/WP4 master dataset.
    """

    required_columns = {
        "company",
        "community",
        "analysis_eligible",
    }

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:

        missing_text = ", ".join(
            sorted(
                missing_columns
            )
        )

        raise ValueError(
            "The master dataset is missing required "
            f"columns: {missing_text}"
        )

    if dataframe.empty:

        raise ValueError(
            "The master dataset is empty."
        )

    if dataframe["company"].isna().any():

        raise ValueError(
            "The master dataset contains missing company "
            "names."
        )

    if dataframe["community"].isna().any():

        raise ValueError(
            "The master dataset contains missing community "
            "assignments."
        )


def validate_temporal_profiles(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate the existing rolling-five-year community
    profiles.
    """

    required_metadata = {
        "company",
        "window",
        "start_year",
        "end_year",
    }

    missing_columns = required_metadata.difference(
        dataframe.columns
    )

    if missing_columns:

        missing_text = ", ".join(
            sorted(
                missing_columns
            )
        )

        raise ValueError(
            "The rolling-five-year profile file is missing "
            f"required columns: {missing_text}"
        )

    if dataframe.empty:

        raise ValueError(
            "The rolling-five-year profile file is empty."
        )

    if dataframe["company"].isna().any():

        raise ValueError(
            "The rolling-five-year profile file contains "
            "missing company names."
        )

    duplicate_windows = (
        dataframe[
            [
                "company",
                "start_year",
                "end_year",
            ]
        ]
        .duplicated(
            keep=False
        )
    )

    if duplicate_windows.any():

        raise ValueError(
            "Duplicate company-window profiles were found "
            "in the rolling-five-year profile file."
        )


def convert_boolean_column(
    values: pd.Series,
    column_name: str,
) -> pd.Series:
    """
    Convert a Boolean-like column safely.
    """

    if pd.api.types.is_bool_dtype(
        values
    ):

        return values.astype(bool)

    converted = (
        values
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
                "yes": True,
                "no": False,
            }
        )
    )

    if converted.isna().any():

        invalid_values = (
            values.loc[
                converted.isna()
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Column '{column_name}' contains values that "
            "cannot be interpreted as Boolean: "
            f"{invalid_values}"
        )

    return converted.astype(bool)


# --------------------------------------------------------
# Community-column handling
# --------------------------------------------------------

def community_sort_key(
    value: object,
) -> tuple:
    """
    Sort community labels numerically where possible and
    lexically otherwise.
    """

    text = str(
        value
    )

    try:

        numeric_value = float(
            text
        )

        return (
            0,
            numeric_value,
            text,
        )

    except ValueError:

        return (
            1,
            text,
            text,
        )


def community_column_name(
    community: object,
) -> str:
    """
    Convert a community label to the canonical probability
    column name used by this robustness script.
    """

    return (
        "community_"
        + str(
            community
        )
    )


def identify_temporal_community_columns(
    dataframe: pd.DataFrame,
) -> list:
    """
    Identify probability columns from the existing WP5
    rolling-five-year profile output.

    Only columns beginning with 'community_' are accepted.
    This avoids accidentally treating summary statistics or
    metadata as fingerprint dimensions.
    """

    community_columns = [
        column
        for column in dataframe.columns
        if (
            str(
                column
            ).startswith(
                "community_"
            )
            and pd.api.types.is_numeric_dtype(
                dataframe[column]
            )
        )
    ]

    if not community_columns:

        raise ValueError(
            "No numeric columns beginning with "
            "'community_' were found in the rolling-five-"
            "year profile file."
        )

    return sorted(
        community_columns,
        key=community_sort_key,
    )


# --------------------------------------------------------
# Static fingerprint construction
# --------------------------------------------------------

def build_static_company_profiles(
    master_data: pd.DataFrame,
) -> tuple[pd.DataFrame, list]:
    """
    Construct company semantic fingerprints from eligible
    master-dataset rows.

    Each fingerprint is the probability distribution of a
    company's eligible songs across the existing community
    assignments.
    """

    working = master_data.copy()

    working["analysis_eligible"] = (
        convert_boolean_column(
            values=working[
                "analysis_eligible"
            ],
            column_name=(
                "analysis_eligible"
            ),
        )
    )

    working = (
        working.loc[
            working[
                "analysis_eligible"
            ]
        ]
        .copy()
    )

    if working.empty:

        raise ValueError(
            "No analysis-eligible rows remain in the master "
            "dataset."
        )

    working["company"] = (
        working["company"]
        .astype(str)
        .str.strip()
    )

    working["community"] = (
        working["community"]
        .astype(str)
        .str.strip()
    )

    if (
        working["company"]
        == ""
    ).any():

        raise ValueError(
            "Eligible master-data rows contain blank "
            "company names."
        )

    if (
        working["community"]
        == ""
    ).any():

        raise ValueError(
            "Eligible master-data rows contain blank "
            "community assignments."
        )

    community_labels = sorted(
        working[
            "community"
        ].unique(),
        key=community_sort_key,
    )

    count_table = (
        working.groupby(
            [
                "company",
                "community",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    count_table = count_table.reindex(
        columns=community_labels,
        fill_value=0,
    )

    count_table.columns = [
        community_column_name(
            community
        )
        for community in count_table.columns
    ]

    probability_table = (
        count_table.div(
            count_table.sum(
                axis=1
            ),
            axis=0,
        )
    )

    company_profiles = (
        probability_table
        .reset_index()
    )

    company_song_counts = (
        working.groupby(
            "company"
        )
        .size()
        .rename(
            "n_songs"
        )
        .reset_index()
    )

    if "artist_normalised" in working.columns:

        company_artist_counts = (
            working.groupby(
                "company"
            )[
                "artist_normalised"
            ]
            .nunique()
            .rename(
                "n_artists"
            )
            .reset_index()
        )

    elif "artist" in working.columns:

        company_artist_counts = (
            working.groupby(
                "company"
            )[
                "artist"
            ]
            .nunique()
            .rename(
                "n_artists"
            )
            .reset_index()
        )

    else:

        company_artist_counts = (
            company_song_counts[
                [
                    "company",
                ]
            ]
            .copy()
        )

        company_artist_counts[
            "n_artists"
        ] = np.nan

    company_profiles = (
        company_profiles
        .merge(
            company_song_counts,
            on="company",
            how="left",
            validate="one_to_one",
        )
        .merge(
            company_artist_counts,
            on="company",
            how="left",
            validate="one_to_one",
        )
    )

    community_columns = [
        column
        for column in company_profiles.columns
        if column.startswith(
            "community_"
        )
    ]

    probability_sums = (
        company_profiles[
            community_columns
        ]
        .sum(
            axis=1
        )
    )

    if not np.allclose(
        probability_sums,
        1.0,
        atol=1e-10,
        rtol=0.0,
    ):

        raise ValueError(
            "Static company fingerprints do not sum to one."
        )

    if (
        company_profiles[
            "company"
        ].nunique()
        < 3
    ):

        raise ValueError(
            "At least three companies are required for "
            "metric-sensitivity analysis."
        )

    company_profiles = (
        company_profiles
        .sort_values(
            "company"
        )
        .reset_index(
            drop=True
        )
    )

    return (
        company_profiles,
        community_columns,
    )


# --------------------------------------------------------
# Temporal profile preparation
# --------------------------------------------------------

def prepare_temporal_profiles(
    temporal_profiles: pd.DataFrame,
    static_community_columns: list,
) -> tuple[pd.DataFrame, list]:
    """
    Prepare the existing rolling-five-year fingerprints and
    align their community dimensions with the static
    fingerprints.
    """

    working = temporal_profiles.copy()

    temporal_community_columns = (
        identify_temporal_community_columns(
            working
        )
    )

    static_communities = set(
        static_community_columns
    )

    temporal_communities = set(
        temporal_community_columns
    )

    missing_from_temporal = sorted(
        static_communities
        - temporal_communities
    )

    additional_temporal = sorted(
        temporal_communities
        - static_communities
    )

    if missing_from_temporal:

        raise ValueError(
            "The rolling-five-year profiles are missing "
            "community columns present in the master "
            f"dataset: {missing_from_temporal}"
        )

    if additional_temporal:

        raise ValueError(
            "The rolling-five-year profiles contain "
            "community columns not represented in the "
            f"master dataset: {additional_temporal}"
        )

    community_columns = list(
        static_community_columns
    )

    working["company"] = (
        working["company"]
        .astype(str)
        .str.strip()
    )

    working["window"] = (
        working["window"]
        .astype(str)
    )

    working["start_year"] = (
        pd.to_numeric(
            working[
                "start_year"
            ],
            errors="raise",
        )
        .astype(int)
    )

    working["end_year"] = (
        pd.to_numeric(
            working[
                "end_year"
            ],
            errors="raise",
        )
        .astype(int)
    )

    for column in community_columns:

        working[column] = (
            pd.to_numeric(
                working[
                    column
                ],
                errors="raise",
            )
            .astype(float)
        )

    if (
        working[
            community_columns
        ]
        < -EPSILON
    ).any().any():

        raise ValueError(
            "The rolling-five-year profiles contain "
            "negative community probabilities."
        )

    probability_sums = (
        working[
            community_columns
        ]
        .sum(
            axis=1
        )
    )

    if (
        probability_sums
        <= EPSILON
    ).any():

        raise ValueError(
            "At least one rolling-five-year fingerprint has "
            "zero total probability."
        )

    working.loc[
        :,
        community_columns,
    ] = (
        working[
            community_columns
        ]
        .div(
            probability_sums,
            axis=0,
        )
    )

    working = (
        working
        .sort_values(
            [
                "company",
                "start_year",
                "end_year",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return (
        working,
        community_columns,
    )


# --------------------------------------------------------
# Distance metrics
# --------------------------------------------------------

def correlation_fallback_required(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
) -> bool:
    """
    Return True if correlation distance is undefined because
    either vector has effectively zero variance.
    """

    variance_a = float(
        np.var(
            vector_a
        )
    )

    variance_b = float(
        np.var(
            vector_b
        )
    )

    return bool(
        variance_a <= EPSILON
        or variance_b <= EPSILON
    )


def calculate_distance(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
    metric: str,
) -> tuple[float, bool]:
    """
    Calculate one distance value.

    Correlation distance is undefined for constant vectors.
    When this occurs, cosine distance is used as a recorded
    numerical fallback rather than silently returning NaN.

    Returns
    -------
    tuple
        Distance value and whether a fallback was used.
    """

    vector_a = np.asarray(
        vector_a,
        dtype=float,
    )

    vector_b = np.asarray(
        vector_b,
        dtype=float,
    )

    fallback_used = False

    if metric == "cosine":

        distance = cosine(
            vector_a,
            vector_b,
        )

    elif metric == "euclidean":

        distance = euclidean(
            vector_a,
            vector_b,
        )

    elif metric == "correlation":

        fallback_used = (
            correlation_fallback_required(
                vector_a,
                vector_b,
            )
        )

        if fallback_used:

            distance = cosine(
                vector_a,
                vector_b,
            )

        else:

            distance = correlation(
                vector_a,
                vector_b,
            )

    else:

        raise ValueError(
            f"Unsupported distance metric: {metric}"
        )

    if not np.isfinite(
        distance
    ):

        raise ValueError(
            f"Non-finite {metric} distance encountered."
        )

    return (
        float(
            distance
        ),
        fallback_used,
    )


# --------------------------------------------------------
# Static company comparisons
# --------------------------------------------------------

def calculate_company_pair_distances(
    company_profiles: pd.DataFrame,
    community_columns: list,
) -> pd.DataFrame:
    """
    Calculate all pairwise static company distances.
    """

    companies = (
        company_profiles[
            "company"
        ]
        .astype(str)
        .tolist()
    )

    profile_lookup = {
        row["company"]: (
            row[
                community_columns
            ]
            .to_numpy(
                dtype=float
            )
        )
        for _, row in company_profiles.iterrows()
    }

    rows = []

    for company_a, company_b in combinations(
        companies,
        2,
    ):

        vector_a = profile_lookup[
            company_a
        ]

        vector_b = profile_lookup[
            company_b
        ]

        row = {
            "company_a": company_a,
            "company_b": company_b,
            "pair_label": (
                f"{company_a}__{company_b}"
            ),
        }

        correlation_fallback = False

        for metric in METRICS:

            (
                distance,
                fallback_used,
            ) = calculate_distance(
                vector_a=vector_a,
                vector_b=vector_b,
                metric=metric,
            )

            row[
                f"{metric}_distance"
            ] = distance

            if (
                metric
                == "correlation"
            ):

                correlation_fallback = (
                    fallback_used
                )

        row[
            "correlation_fallback"
        ] = correlation_fallback

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def build_company_rankings(
    company_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rank companies by average distance from all other
    companies under each metric.
    """

    companies = sorted(
        set(
            company_pairs[
                "company_a"
            ]
        ).union(
            set(
                company_pairs[
                    "company_b"
                ]
            )
        )
    )

    rows = []

    for metric in METRICS:

        distance_column = (
            f"{metric}_distance"
        )

        for company in companies:

            company_distances = (
                company_pairs.loc[
                    (
                        company_pairs[
                            "company_a"
                        ] == company
                    )
                    |
                    (
                        company_pairs[
                            "company_b"
                        ] == company
                    ),
                    distance_column,
                ]
            )

            rows.append(
                {
                    "metric": metric,
                    "company": company,
                    "n_company_pairs": int(
                        len(
                            company_distances
                        )
                    ),
                    "mean_pairwise_distance": float(
                        company_distances.mean()
                    ),
                    "median_pairwise_distance": float(
                        company_distances.median()
                    ),
                    "std_pairwise_distance": float(
                        company_distances.std(
                            ddof=1
                        )
                    ),
                    "minimum_pairwise_distance": float(
                        company_distances.min()
                    ),
                    "maximum_pairwise_distance": float(
                        company_distances.max()
                    ),
                }
            )

    rankings = pd.DataFrame(
        rows
    )

    rankings[
        "distinctiveness_rank"
    ] = (
        rankings.groupby(
            "metric"
        )[
            "mean_pairwise_distance"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    return (
        rankings
        .sort_values(
            [
                "metric",
                "distinctiveness_rank",
                "company",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# --------------------------------------------------------
# Temporal drift comparisons
# --------------------------------------------------------

def calculate_temporal_transitions(
    temporal_profiles: pd.DataFrame,
    community_columns: list,
) -> pd.DataFrame:
    """
    Recalculate consecutive rolling-five-year semantic drift
    under all three distance metrics.
    """

    rows = []

    for company, company_df in (
        temporal_profiles.groupby(
            "company",
            sort=True,
        )
    ):

        company_df = (
            company_df
            .sort_values(
                [
                    "start_year",
                    "end_year",
                ]
            )
            .reset_index(
                drop=True
            )
        )

        for row_index in range(
            1,
            len(
                company_df
            ),
        ):

            previous_row = (
                company_df.iloc[
                    row_index - 1
                ]
            )

            current_row = (
                company_df.iloc[
                    row_index
                ]
            )

            from_start = int(
                previous_row[
                    "start_year"
                ]
            )

            to_start = int(
                current_row[
                    "start_year"
                ]
            )

            year_gap = (
                to_start
                - from_start
            )

            if year_gap != 1:

                continue

            vector_previous = (
                previous_row[
                    community_columns
                ]
                .to_numpy(
                    dtype=float
                )
            )

            vector_current = (
                current_row[
                    community_columns
                ]
                .to_numpy(
                    dtype=float
                )
            )

            row = {
                "company": company,
                "transition": (
                    str(
                        previous_row[
                            "window"
                        ]
                    )
                    + " -> "
                    + str(
                        current_row[
                            "window"
                        ]
                    )
                ),
                "from_window": str(
                    previous_row[
                        "window"
                    ]
                ),
                "to_window": str(
                    current_row[
                        "window"
                    ]
                ),
                "from_start": (
                    from_start
                ),
                "to_start": (
                    to_start
                ),
                "from_end": int(
                    previous_row[
                        "end_year"
                    ]
                ),
                "to_end": int(
                    current_row[
                        "end_year"
                    ]
                ),
                "year_gap": (
                    year_gap
                ),
                "is_consecutive": True,
            }

            correlation_fallback = False

            for metric in METRICS:

                (
                    distance,
                    fallback_used,
                ) = calculate_distance(
                    vector_a=vector_previous,
                    vector_b=vector_current,
                    metric=metric,
                )

                row[
                    f"{metric}_distance"
                ] = distance

                if (
                    metric
                    == "correlation"
                ):

                    correlation_fallback = (
                        fallback_used
                    )

            row[
                "correlation_fallback"
            ] = correlation_fallback

            rows.append(
                row
            )

    transitions = pd.DataFrame(
        rows
    )

    if transitions.empty:

        raise ValueError(
            "No consecutive rolling-five-year transitions "
            "were available for metric sensitivity."
        )

    return (
        transitions
        .sort_values(
            [
                "company",
                "from_start",
                "to_start",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def build_temporal_rankings(
    temporal_transitions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarise and rank company temporal drift under each
    distance metric.
    """

    ranking_frames = []

    for metric in METRICS:

        distance_column = (
            f"{metric}_distance"
        )

        company_summary = (
            temporal_transitions.groupby(
                "company",
                sort=True,
            )[
                distance_column
            ]
            .agg(
                n_transitions="count",
                mean_drift="mean",
                median_drift="median",
                std_drift="std",
                minimum_drift="min",
                maximum_drift="max",
            )
            .reset_index()
        )

        company_summary.insert(
            0,
            "metric",
            metric,
        )

        company_summary[
            "temporal_drift_rank"
        ] = (
            company_summary[
                "mean_drift"
            ]
            .rank(
                method="min",
                ascending=False,
            )
            .astype(int)
        )

        company_summary[
            "temporal_stability_rank"
        ] = (
            company_summary[
                "mean_drift"
            ]
            .rank(
                method="min",
                ascending=True,
            )
            .astype(int)
        )

        ranking_frames.append(
            company_summary
        )

    rankings = pd.concat(
        ranking_frames,
        ignore_index=True,
    )

    return (
        rankings
        .sort_values(
            [
                "metric",
                "temporal_drift_rank",
                "company",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# --------------------------------------------------------
# Statistical agreement helpers
# --------------------------------------------------------

def calculate_spearman(
    values_a: pd.Series,
    values_b: pd.Series,
) -> tuple[float, float]:
    """
    Calculate Spearman correlation and p-value.
    """

    result = spearmanr(
        values_a,
        values_b,
    )

    return (
        float(
            result.statistic
        ),
        float(
            result.pvalue
        ),
    )


def calculate_kendall(
    values_a: pd.Series,
    values_b: pd.Series,
) -> tuple[float, float]:
    """
    Calculate Kendall tau and p-value.
    """

    result = kendalltau(
        values_a,
        values_b,
    )

    return (
        float(
            result.statistic
        ),
        float(
            result.pvalue
        ),
    )


def calculate_pearson(
    values_a: pd.Series,
    values_b: pd.Series,
) -> tuple[float, float]:
    """
    Calculate Pearson correlation and p-value.
    """

    result = pearsonr(
        values_a,
        values_b,
    )

    return (
        float(
            result.statistic
        ),
        float(
            result.pvalue
        ),
    )


def min_max_scale(
    values: pd.Series,
) -> pd.Series:
    """
    Min-max scale a distance vector for cross-metric
    magnitude comparison.
    """

    numeric_values = (
        pd.to_numeric(
            values,
            errors="raise",
        )
        .astype(float)
    )

    minimum_value = float(
        numeric_values.min()
    )

    maximum_value = float(
        numeric_values.max()
    )

    value_range = (
        maximum_value
        - minimum_value
    )

    if value_range <= EPSILON:

        return pd.Series(
            0.0,
            index=numeric_values.index,
            dtype=float,
        )

    return (
        numeric_values
        - minimum_value
    ) / value_range


# --------------------------------------------------------
# Pairwise distance-matrix agreement
# --------------------------------------------------------

def compare_pairwise_distances(
    company_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare pairwise company distances across metrics.
    """

    rows = []

    for metric_a, metric_b in combinations(
        METRICS,
        2,
    ):

        values_a = (
            company_pairs[
                f"{metric_a}_distance"
            ]
        )

        values_b = (
            company_pairs[
                f"{metric_b}_distance"
            ]
        )

        (
            spearman_r,
            spearman_p,
        ) = calculate_spearman(
            values_a,
            values_b,
        )

        (
            kendall_tau,
            kendall_p,
        ) = calculate_kendall(
            values_a,
            values_b,
        )

        (
            pearson_r,
            pearson_p,
        ) = calculate_pearson(
            values_a,
            values_b,
        )

        scaled_a = min_max_scale(
            values_a
        )

        scaled_b = min_max_scale(
            values_b
        )

        rows.append(
            {
                "analysis_type": (
                    "pairwise_fingerprint_distances"
                ),
                "metric_a": metric_a,
                "metric_b": metric_b,
                "n_observations": int(
                    len(
                        company_pairs
                    )
                ),
                "spearman_r": (
                    spearman_r
                ),
                "spearman_p": (
                    spearman_p
                ),
                "kendall_tau": (
                    kendall_tau
                ),
                "kendall_p": (
                    kendall_p
                ),
                "pearson_r": (
                    pearson_r
                ),
                "pearson_p": (
                    pearson_p
                ),
                "mean_absolute_scaled_difference": float(
                    (
                        scaled_a
                        - scaled_b
                    )
                    .abs()
                    .mean()
                ),
                "maximum_absolute_scaled_difference": float(
                    (
                        scaled_a
                        - scaled_b
                    )
                    .abs()
                    .max()
                ),
                "exact_rank_agreement": np.nan,
                "mean_absolute_rank_difference": np.nan,
                "maximum_rank_difference": np.nan,
                "top_rank_agreement": np.nan,
                "bottom_rank_agreement": np.nan,
            }
        )

    return pd.DataFrame(
        rows
    )


# --------------------------------------------------------
# Ranking agreement
# --------------------------------------------------------

def compare_metric_rankings(
    rankings: pd.DataFrame,
    rank_column: str,
    analysis_type: str,
) -> pd.DataFrame:
    """
    Compare company rankings between all metric pairs.
    """

    rows = []

    for metric_a, metric_b in combinations(
        METRICS,
        2,
    ):

        ranking_a = (
            rankings.loc[
                rankings[
                    "metric"
                ] == metric_a,
                [
                    "company",
                    rank_column,
                ],
            ]
            .rename(
                columns={
                    rank_column: (
                        "rank_a"
                    )
                }
            )
        )

        ranking_b = (
            rankings.loc[
                rankings[
                    "metric"
                ] == metric_b,
                [
                    "company",
                    rank_column,
                ],
            ]
            .rename(
                columns={
                    rank_column: (
                        "rank_b"
                    )
                }
            )
        )

        comparison = ranking_a.merge(
            ranking_b,
            on="company",
            how="inner",
            validate="one_to_one",
        )

        if comparison.empty:

            raise ValueError(
                "No matching companies were available for "
                f"{metric_a} and {metric_b} ranking "
                "comparison."
            )

        (
            spearman_r,
            spearman_p,
        ) = calculate_spearman(
            comparison[
                "rank_a"
            ],
            comparison[
                "rank_b"
            ],
        )

        (
            kendall_tau,
            kendall_p,
        ) = calculate_kendall(
            comparison[
                "rank_a"
            ],
            comparison[
                "rank_b"
            ],
        )

        rank_differences = (
            comparison[
                "rank_a"
            ]
            - comparison[
                "rank_b"
            ]
        ).abs()

        top_company_a = str(
            comparison.loc[
                comparison[
                    "rank_a"
                ].idxmin(),
                "company",
            ]
        )

        top_company_b = str(
            comparison.loc[
                comparison[
                    "rank_b"
                ].idxmin(),
                "company",
            ]
        )

        bottom_company_a = str(
            comparison.loc[
                comparison[
                    "rank_a"
                ].idxmax(),
                "company",
            ]
        )

        bottom_company_b = str(
            comparison.loc[
                comparison[
                    "rank_b"
                ].idxmax(),
                "company",
            ]
        )

        rows.append(
            {
                "analysis_type": (
                    analysis_type
                ),
                "metric_a": metric_a,
                "metric_b": metric_b,
                "n_observations": int(
                    len(
                        comparison
                    )
                ),
                "spearman_r": (
                    spearman_r
                ),
                "spearman_p": (
                    spearman_p
                ),
                "kendall_tau": (
                    kendall_tau
                ),
                "kendall_p": (
                    kendall_p
                ),
                "pearson_r": np.nan,
                "pearson_p": np.nan,
                "mean_absolute_scaled_difference": np.nan,
                "maximum_absolute_scaled_difference": np.nan,
                "exact_rank_agreement": float(
                    (
                        comparison[
                            "rank_a"
                        ]
                        == comparison[
                            "rank_b"
                        ]
                    )
                    .mean()
                ),
                "mean_absolute_rank_difference": float(
                    rank_differences.mean()
                ),
                "maximum_rank_difference": int(
                    rank_differences.max()
                ),
                "top_rank_agreement": bool(
                    top_company_a
                    == top_company_b
                ),
                "bottom_rank_agreement": bool(
                    bottom_company_a
                    == bottom_company_b
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# --------------------------------------------------------
# Baseline-metric comparison
# --------------------------------------------------------

def build_baseline_rank_comparison(
    rankings: pd.DataFrame,
    rank_column: str,
    value_column: str,
    comparison_name: str,
) -> pd.DataFrame:
    """
    Create a company-level comparison against the cosine
    baseline for direct manuscript inspection.
    """

    baseline = (
        rankings.loc[
            rankings[
                "metric"
            ] == BASELINE_METRIC,
            [
                "company",
                rank_column,
                value_column,
            ],
        ]
        .rename(
            columns={
                rank_column: (
                    "cosine_rank"
                ),
                value_column: (
                    "cosine_value"
                ),
            }
        )
    )

    comparison_frames = []

    for metric in METRICS:

        metric_data = (
            rankings.loc[
                rankings[
                    "metric"
                ] == metric,
                [
                    "company",
                    rank_column,
                    value_column,
                ],
            ]
            .rename(
                columns={
                    rank_column: (
                        "metric_rank"
                    ),
                    value_column: (
                        "metric_value"
                    ),
                }
            )
        )

        comparison = baseline.merge(
            metric_data,
            on="company",
            how="inner",
            validate="one_to_one",
        )

        comparison.insert(
            0,
            "comparison",
            comparison_name,
        )

        comparison.insert(
            1,
            "metric",
            metric,
        )

        comparison[
            "absolute_rank_difference"
        ] = (
            comparison[
                "cosine_rank"
            ]
            - comparison[
                "metric_rank"
            ]
        ).abs()

        comparison_frames.append(
            comparison
        )

    return pd.concat(
        comparison_frames,
        ignore_index=True,
    )


# --------------------------------------------------------
# Robustness classification
# --------------------------------------------------------

def classify_pairwise_robustness(
    minimum_spearman: float,
) -> str:
    """
    Classify pairwise matrix agreement.
    """

    if minimum_spearman >= 0.90:

        return "highly_robust"

    if minimum_spearman >= 0.75:

        return "robust"

    if minimum_spearman >= 0.50:

        return "moderately_robust"

    return "metric_sensitive"


def classify_ranking_robustness(
    minimum_spearman: float,
    top_agreement_rate: float,
) -> str:
    """
    Classify company-ranking agreement.
    """

    if (
        minimum_spearman >= 0.90
        and top_agreement_rate >= 1.0
    ):

        return "highly_robust"

    if (
        minimum_spearman >= 0.75
        and top_agreement_rate >= (
            2.0 / 3.0
        )
    ):

        return "robust"

    if minimum_spearman >= 0.50:

        return "moderately_robust"

    return "metric_sensitive"

def get_baseline_agreement_row(
    agreement: pd.DataFrame,
    analysis_type: str,
    alternative_metric: str,
) -> pd.Series:
    """
    Retrieve an agreement result comparing the cosine
    baseline with one alternative metric.

    The function accepts either metric ordering, although
    the current agreement table normally stores cosine as
    metric_a.
    """

    matching_rows = agreement.loc[
        (
            agreement[
                "analysis_type"
            ] == analysis_type
        )
        &
        (
            (
                (
                    agreement[
                        "metric_a"
                    ] == BASELINE_METRIC
                )
                &
                (
                    agreement[
                        "metric_b"
                    ] == alternative_metric
                )
            )
            |
            (
                (
                    agreement[
                        "metric_b"
                    ] == BASELINE_METRIC
                )
                &
                (
                    agreement[
                        "metric_a"
                    ] == alternative_metric
                )
            )
        )
    ]

    if len(
        matching_rows
    ) != 1:

        raise ValueError(
            "Expected exactly one agreement row for "
            f"analysis '{analysis_type}' comparing "
            f"'{BASELINE_METRIC}' with "
            f"'{alternative_metric}', but found "
            f"{len(matching_rows)}."
        )

    return matching_rows.iloc[
        0
    ]


def calculate_rank_set_agreement(
    rankings: pd.DataFrame,
    rank_column: str,
    alternative_metric: str,
    set_size: int = 2,
) -> dict:
    """
    Compare top-k and bottom-k company membership between
    cosine and an alternative metric.

    This captures stable substantive groupings even when
    companies exchange positions within a group.
    """

    baseline_rankings = (
        rankings.loc[
            rankings[
                "metric"
            ] == BASELINE_METRIC,
            [
                "company",
                rank_column,
            ],
        ]
        .sort_values(
            [
                rank_column,
                "company",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    alternative_rankings = (
        rankings.loc[
            rankings[
                "metric"
            ] == alternative_metric,
            [
                "company",
                rank_column,
            ],
        ]
        .sort_values(
            [
                rank_column,
                "company",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    baseline_companies = set(
        baseline_rankings[
            "company"
        ]
    )

    alternative_companies = set(
        alternative_rankings[
            "company"
        ]
    )

    if (
        baseline_companies
        != alternative_companies
    ):

        raise ValueError(
            "Company membership differs between cosine and "
            f"{alternative_metric} rankings."
        )

    n_companies = len(
        baseline_rankings
    )

    if set_size < 1:

        raise ValueError(
            "set_size must be at least one."
        )

    if set_size > n_companies:

        raise ValueError(
            "set_size cannot exceed the number of "
            "companies."
        )

    baseline_top = set(
        baseline_rankings.head(
            set_size
        )[
            "company"
        ]
    )

    alternative_top = set(
        alternative_rankings.head(
            set_size
        )[
            "company"
        ]
    )

    baseline_bottom = set(
        baseline_rankings.tail(
            set_size
        )[
            "company"
        ]
    )

    alternative_bottom = set(
        alternative_rankings.tail(
            set_size
        )[
            "company"
        ]
    )

    top_overlap = len(
        baseline_top.intersection(
            alternative_top
        )
    )

    bottom_overlap = len(
        baseline_bottom.intersection(
            alternative_bottom
        )
    )

    return {
        "set_size": int(
            set_size
        ),
        "baseline_top_set": "|".join(
            sorted(
                baseline_top
            )
        ),
        "alternative_top_set": "|".join(
            sorted(
                alternative_top
            )
        ),
        "top_set_overlap": int(
            top_overlap
        ),
        "top_set_agreement": float(
            top_overlap
            / set_size
        ),
        "baseline_bottom_set": "|".join(
            sorted(
                baseline_bottom
            )
        ),
        "alternative_bottom_set": "|".join(
            sorted(
                alternative_bottom
            )
        ),
        "bottom_set_overlap": int(
            bottom_overlap
        ),
        "bottom_set_agreement": float(
            bottom_overlap
            / set_size
        ),
    }


def classify_baseline_distance_robustness(
    spearman_r: float,
    kendall_tau: float,
) -> str:
    """
    Classify agreement between cosine and one alternative
    metric for pairwise fingerprint distances.
    """

    if (
        spearman_r >= 0.90
        and kendall_tau >= 0.80
    ):

        return "highly_robust"

    if (
        spearman_r >= 0.75
        and kendall_tau >= 0.60
    ):

        return "robust"

    if spearman_r >= 0.50:

        return "moderately_robust"

    return "metric_sensitive"


def classify_baseline_ranking_robustness(
    spearman_r: float,
    maximum_rank_difference: float,
    top_set_agreement: float,
    bottom_set_agreement: float,
) -> str:
    """
    Classify ranking robustness while recognising stable
    company-group membership.

    A ranking can retain its substantive top and bottom
    groups even when companies exchange positions within
    those groups.
    """

    if (
        spearman_r >= 0.90
        and maximum_rank_difference == 0
        and top_set_agreement == 1.0
        and bottom_set_agreement == 1.0
    ):

        return "highly_robust"

    if (
        spearman_r >= 0.75
        and maximum_rank_difference <= 1
        and top_set_agreement == 1.0
        and bottom_set_agreement == 1.0
    ):

        return "robust"

    if (
        maximum_rank_difference <= 1
        and top_set_agreement == 1.0
        and bottom_set_agreement == 1.0
    ):

        return "broad_grouping_robust"

    if spearman_r >= 0.50:

        return "moderately_robust"

    return "metric_sensitive"
# --------------------------------------------------------
# Publication summary
# --------------------------------------------------------

def build_publication_summary(
    agreement: pd.DataFrame,
    company_pairs: pd.DataFrame,
    company_rankings: pd.DataFrame,
    temporal_transitions: pd.DataFrame,
    temporal_rankings: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a manuscript-facing metric-sensitivity summary.

    Cosine is treated as the baseline metric. Euclidean and
    correlation distance are evaluated separately rather
    than collapsing all comparisons into a single
    worst-case result.

    The complete all-pairs agreement results remain
    available in metric_sensitivity_agreement.csv.
    """

    summary_rows = []

    # ----------------------------------------------------
    # Static pairwise fingerprint distances
    # ----------------------------------------------------

    for alternative_metric in [
        "euclidean",
        "correlation",
    ]:

        agreement_row = (
            get_baseline_agreement_row(
                agreement=agreement,
                analysis_type=(
                    "pairwise_fingerprint_distances"
                ),
                alternative_metric=(
                    alternative_metric
                ),
            )
        )

        spearman_r = float(
            agreement_row[
                "spearman_r"
            ]
        )

        kendall_tau = float(
            agreement_row[
                "kendall_tau"
            ]
        )

        summary_rows.append(
            {
                "analysis": (
                    "Pairwise fingerprint distances"
                ),
                "baseline_metric": (
                    BASELINE_METRIC
                ),
                "alternative_metric": (
                    alternative_metric
                ),
                "earlier_finding_tested": (
                    "Relative company fingerprint "
                    "similarity"
                ),
                "n_observations": int(
                    agreement_row[
                        "n_observations"
                    ]
                ),
                "spearman_r": (
                    spearman_r
                ),
                "spearman_p": float(
                    agreement_row[
                        "spearman_p"
                    ]
                ),
                "kendall_tau": (
                    kendall_tau
                ),
                "kendall_p": float(
                    agreement_row[
                        "kendall_p"
                    ]
                ),
                "pearson_r": float(
                    agreement_row[
                        "pearson_r"
                    ]
                ),
                "pearson_p": float(
                    agreement_row[
                        "pearson_p"
                    ]
                ),
                "exact_rank_agreement": np.nan,
                "mean_absolute_rank_difference": np.nan,
                "maximum_rank_difference": np.nan,
                "top_rank_agreement": np.nan,
                "bottom_rank_agreement": np.nan,
                "top_two_set_agreement": np.nan,
                "bottom_two_set_agreement": np.nan,
                "cosine_top_two": "",
                "alternative_top_two": "",
                "cosine_bottom_two": "",
                "alternative_bottom_two": "",
                "mean_absolute_scaled_difference": float(
                    agreement_row[
                        "mean_absolute_scaled_difference"
                    ]
                ),
                "maximum_absolute_scaled_difference": float(
                    agreement_row[
                        "maximum_absolute_scaled_difference"
                    ]
                ),
                "conclusion": (
                    classify_baseline_distance_robustness(
                        spearman_r=spearman_r,
                        kendall_tau=kendall_tau,
                    )
                ),
            }
        )

    # ----------------------------------------------------
    # Static company distinctiveness rankings
    # ----------------------------------------------------

    for alternative_metric in [
        "euclidean",
        "correlation",
    ]:

        agreement_row = (
            get_baseline_agreement_row(
                agreement=agreement,
                analysis_type=(
                    "company_distinctiveness_rankings"
                ),
                alternative_metric=(
                    alternative_metric
                ),
            )
        )

        set_agreement = (
            calculate_rank_set_agreement(
                rankings=company_rankings,
                rank_column=(
                    "distinctiveness_rank"
                ),
                alternative_metric=(
                    alternative_metric
                ),
                set_size=2,
            )
        )

        spearman_r = float(
            agreement_row[
                "spearman_r"
            ]
        )

        maximum_rank_difference = float(
            agreement_row[
                "maximum_rank_difference"
            ]
        )

        summary_rows.append(
            {
                "analysis": (
                    "Company distinctiveness ranking"
                ),
                "baseline_metric": (
                    BASELINE_METRIC
                ),
                "alternative_metric": (
                    alternative_metric
                ),
                "earlier_finding_tested": (
                    "Ranking of companies by semantic "
                    "distinctiveness"
                ),
                "n_observations": int(
                    agreement_row[
                        "n_observations"
                    ]
                ),
                "spearman_r": (
                    spearman_r
                ),
                "spearman_p": float(
                    agreement_row[
                        "spearman_p"
                    ]
                ),
                "kendall_tau": float(
                    agreement_row[
                        "kendall_tau"
                    ]
                ),
                "kendall_p": float(
                    agreement_row[
                        "kendall_p"
                    ]
                ),
                "pearson_r": np.nan,
                "pearson_p": np.nan,
                "exact_rank_agreement": float(
                    agreement_row[
                        "exact_rank_agreement"
                    ]
                ),
                "mean_absolute_rank_difference": float(
                    agreement_row[
                        "mean_absolute_rank_difference"
                    ]
                ),
                "maximum_rank_difference": (
                    maximum_rank_difference
                ),
                "top_rank_agreement": bool(
                    agreement_row[
                        "top_rank_agreement"
                    ]
                ),
                "bottom_rank_agreement": bool(
                    agreement_row[
                        "bottom_rank_agreement"
                    ]
                ),
                "top_two_set_agreement": float(
                    set_agreement[
                        "top_set_agreement"
                    ]
                ),
                "bottom_two_set_agreement": float(
                    set_agreement[
                        "bottom_set_agreement"
                    ]
                ),
                "cosine_top_two": (
                    set_agreement[
                        "baseline_top_set"
                    ]
                ),
                "alternative_top_two": (
                    set_agreement[
                        "alternative_top_set"
                    ]
                ),
                "cosine_bottom_two": (
                    set_agreement[
                        "baseline_bottom_set"
                    ]
                ),
                "alternative_bottom_two": (
                    set_agreement[
                        "alternative_bottom_set"
                    ]
                ),
                "mean_absolute_scaled_difference": np.nan,
                "maximum_absolute_scaled_difference": np.nan,
                "conclusion": (
                    classify_baseline_ranking_robustness(
                        spearman_r=spearman_r,
                        maximum_rank_difference=(
                            maximum_rank_difference
                        ),
                        top_set_agreement=float(
                            set_agreement[
                                "top_set_agreement"
                            ]
                        ),
                        bottom_set_agreement=float(
                            set_agreement[
                                "bottom_set_agreement"
                            ]
                        ),
                    )
                ),
            }
        )

    # ----------------------------------------------------
    # Temporal drift rankings
    # ----------------------------------------------------

    for alternative_metric in [
        "euclidean",
        "correlation",
    ]:

        agreement_row = (
            get_baseline_agreement_row(
                agreement=agreement,
                analysis_type=(
                    "temporal_drift_rankings"
                ),
                alternative_metric=(
                    alternative_metric
                ),
            )
        )

        set_agreement = (
            calculate_rank_set_agreement(
                rankings=temporal_rankings,
                rank_column=(
                    "temporal_drift_rank"
                ),
                alternative_metric=(
                    alternative_metric
                ),
                set_size=2,
            )
        )

        spearman_r = float(
            agreement_row[
                "spearman_r"
            ]
        )

        maximum_rank_difference = float(
            agreement_row[
                "maximum_rank_difference"
            ]
        )

        summary_rows.append(
            {
                "analysis": (
                    "Temporal drift ranking"
                ),
                "baseline_metric": (
                    BASELINE_METRIC
                ),
                "alternative_metric": (
                    alternative_metric
                ),
                "earlier_finding_tested": (
                    "Ranking of companies by temporal "
                    "semantic change"
                ),
                "n_observations": int(
                    agreement_row[
                        "n_observations"
                    ]
                ),
                "spearman_r": (
                    spearman_r
                ),
                "spearman_p": float(
                    agreement_row[
                        "spearman_p"
                    ]
                ),
                "kendall_tau": float(
                    agreement_row[
                        "kendall_tau"
                    ]
                ),
                "kendall_p": float(
                    agreement_row[
                        "kendall_p"
                    ]
                ),
                "pearson_r": np.nan,
                "pearson_p": np.nan,
                "exact_rank_agreement": float(
                    agreement_row[
                        "exact_rank_agreement"
                    ]
                ),
                "mean_absolute_rank_difference": float(
                    agreement_row[
                        "mean_absolute_rank_difference"
                    ]
                ),
                "maximum_rank_difference": (
                    maximum_rank_difference
                ),
                "top_rank_agreement": bool(
                    agreement_row[
                        "top_rank_agreement"
                    ]
                ),
                "bottom_rank_agreement": bool(
                    agreement_row[
                        "bottom_rank_agreement"
                    ]
                ),
                "top_two_set_agreement": float(
                    set_agreement[
                        "top_set_agreement"
                    ]
                ),
                "bottom_two_set_agreement": float(
                    set_agreement[
                        "bottom_set_agreement"
                    ]
                ),
                "cosine_top_two": (
                    set_agreement[
                        "baseline_top_set"
                    ]
                ),
                "alternative_top_two": (
                    set_agreement[
                        "alternative_top_set"
                    ]
                ),
                "cosine_bottom_two": (
                    set_agreement[
                        "baseline_bottom_set"
                    ]
                ),
                "alternative_bottom_two": (
                    set_agreement[
                        "alternative_bottom_set"
                    ]
                ),
                "mean_absolute_scaled_difference": np.nan,
                "maximum_absolute_scaled_difference": np.nan,
                "conclusion": (
                    classify_baseline_ranking_robustness(
                        spearman_r=spearman_r,
                        maximum_rank_difference=(
                            maximum_rank_difference
                        ),
                        top_set_agreement=float(
                            set_agreement[
                                "top_set_agreement"
                            ]
                        ),
                        bottom_set_agreement=float(
                            set_agreement[
                                "bottom_set_agreement"
                            ]
                        ),
                    )
                ),
            }
        )

    # ----------------------------------------------------
    # Temporal stability rankings
    # ----------------------------------------------------

    for alternative_metric in [
        "euclidean",
        "correlation",
    ]:

        agreement_row = (
            get_baseline_agreement_row(
                agreement=agreement,
                analysis_type=(
                    "temporal_stability_rankings"
                ),
                alternative_metric=(
                    alternative_metric
                ),
            )
        )

        set_agreement = (
            calculate_rank_set_agreement(
                rankings=temporal_rankings,
                rank_column=(
                    "temporal_stability_rank"
                ),
                alternative_metric=(
                    alternative_metric
                ),
                set_size=2,
            )
        )

        spearman_r = float(
            agreement_row[
                "spearman_r"
            ]
        )

        maximum_rank_difference = float(
            agreement_row[
                "maximum_rank_difference"
            ]
        )

        summary_rows.append(
            {
                "analysis": (
                    "Temporal stability ranking"
                ),
                "baseline_metric": (
                    BASELINE_METRIC
                ),
                "alternative_metric": (
                    alternative_metric
                ),
                "earlier_finding_tested": (
                    "Ranking of companies by temporal "
                    "semantic stability"
                ),
                "n_observations": int(
                    agreement_row[
                        "n_observations"
                    ]
                ),
                "spearman_r": (
                    spearman_r
                ),
                "spearman_p": float(
                    agreement_row[
                        "spearman_p"
                    ]
                ),
                "kendall_tau": float(
                    agreement_row[
                        "kendall_tau"
                    ]
                ),
                "kendall_p": float(
                    agreement_row[
                        "kendall_p"
                    ]
                ),
                "pearson_r": np.nan,
                "pearson_p": np.nan,
                "exact_rank_agreement": float(
                    agreement_row[
                        "exact_rank_agreement"
                    ]
                ),
                "mean_absolute_rank_difference": float(
                    agreement_row[
                        "mean_absolute_rank_difference"
                    ]
                ),
                "maximum_rank_difference": (
                    maximum_rank_difference
                ),
                "top_rank_agreement": bool(
                    agreement_row[
                        "top_rank_agreement"
                    ]
                ),
                "bottom_rank_agreement": bool(
                    agreement_row[
                        "bottom_rank_agreement"
                    ]
                ),
                "top_two_set_agreement": float(
                    set_agreement[
                        "top_set_agreement"
                    ]
                ),
                "bottom_two_set_agreement": float(
                    set_agreement[
                        "bottom_set_agreement"
                    ]
                ),
                "cosine_top_two": (
                    set_agreement[
                        "baseline_top_set"
                    ]
                ),
                "alternative_top_two": (
                    set_agreement[
                        "alternative_top_set"
                    ]
                ),
                "cosine_bottom_two": (
                    set_agreement[
                        "baseline_bottom_set"
                    ]
                ),
                "alternative_bottom_two": (
                    set_agreement[
                        "alternative_bottom_set"
                    ]
                ),
                "mean_absolute_scaled_difference": np.nan,
                "maximum_absolute_scaled_difference": np.nan,
                "conclusion": (
                    classify_baseline_ranking_robustness(
                        spearman_r=spearman_r,
                        maximum_rank_difference=(
                            maximum_rank_difference
                        ),
                        top_set_agreement=float(
                            set_agreement[
                                "top_set_agreement"
                            ]
                        ),
                        bottom_set_agreement=float(
                            set_agreement[
                                "bottom_set_agreement"
                            ]
                        ),
                    )
                ),
            }
        )

    # ----------------------------------------------------
    # Correlation numerical stability
    # ----------------------------------------------------

    static_fallbacks = int(
        company_pairs[
            "correlation_fallback"
        ].sum()
    )

    temporal_fallbacks = int(
        temporal_transitions[
            "correlation_fallback"
        ].sum()
    )

    total_fallbacks = (
        static_fallbacks
        + temporal_fallbacks
    )

    summary_rows.append(
        {
            "analysis": (
                "Correlation-distance numerical stability"
            ),
            "baseline_metric": "",
            "alternative_metric": (
                "correlation"
            ),
            "earlier_finding_tested": (
                "Numerical availability of correlation "
                "distance"
            ),
            "n_observations": int(
                len(
                    company_pairs
                )
                + len(
                    temporal_transitions
                )
            ),
            "spearman_r": np.nan,
            "spearman_p": np.nan,
            "kendall_tau": np.nan,
            "kendall_p": np.nan,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "exact_rank_agreement": np.nan,
            "mean_absolute_rank_difference": np.nan,
            "maximum_rank_difference": np.nan,
            "top_rank_agreement": np.nan,
            "bottom_rank_agreement": np.nan,
            "top_two_set_agreement": np.nan,
            "bottom_two_set_agreement": np.nan,
            "cosine_top_two": "",
            "alternative_top_two": "",
            "cosine_bottom_two": "",
            "alternative_bottom_two": "",
            "mean_absolute_scaled_difference": np.nan,
            "maximum_absolute_scaled_difference": float(
                total_fallbacks
            ),
            "conclusion": (
                "no_fallback_required"
                if total_fallbacks == 0
                else "fallbacks_recorded"
            ),
        }
    )

    publication_summary = pd.DataFrame(
        summary_rows
    )

    return publication_summary[
        [
            "analysis",
            "baseline_metric",
            "alternative_metric",
            "earlier_finding_tested",
            "n_observations",
            "spearman_r",
            "spearman_p",
            "kendall_tau",
            "kendall_p",
            "pearson_r",
            "pearson_p",
            "exact_rank_agreement",
            "mean_absolute_rank_difference",
            "maximum_rank_difference",
            "top_rank_agreement",
            "bottom_rank_agreement",
            "top_two_set_agreement",
            "bottom_two_set_agreement",
            "cosine_top_two",
            "alternative_top_two",
            "cosine_bottom_two",
            "alternative_bottom_two",
            "mean_absolute_scaled_difference",
            "maximum_absolute_scaled_difference",
            "conclusion",
        ]
    ]

# --------------------------------------------------------
# Main
# --------------------------------------------------------

def main() -> None:
    """
    Run WP6 similarity-metric sensitivity analysis.
    """

    print("=" * 70)
    print("WP6 : SIMILARITY METRIC SENSITIVITY")
    print("=" * 70)

    if not MASTER_DATA_FILE.exists():

        raise FileNotFoundError(
            "Required WP3/WP4 master dataset not found:\n"
            f"{MASTER_DATA_FILE}"
        )

    if not TEMPORAL_PROFILE_FILE.exists():

        raise FileNotFoundError(
            "Required WP5 rolling-five-year profile file "
            f"not found:\n{TEMPORAL_PROFILE_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------
    # Load exact pipeline inputs
    # ----------------------------------------------------

    master_data = pd.read_parquet(
        MASTER_DATA_FILE
    )

    temporal_raw = pd.read_parquet(
        TEMPORAL_PROFILE_FILE
    )

    validate_master_data(
        master_data
    )

    validate_temporal_profiles(
        temporal_raw
    )

    # ----------------------------------------------------
    # Construct static company fingerprints
    # ----------------------------------------------------

    (
        company_profiles,
        static_community_columns,
    ) = build_static_company_profiles(
        master_data
    )

    # ----------------------------------------------------
    # Prepare existing rolling-five-year fingerprints
    # ----------------------------------------------------

    (
        temporal_profiles,
        temporal_community_columns,
    ) = prepare_temporal_profiles(
        temporal_profiles=temporal_raw,
        static_community_columns=(
            static_community_columns
        ),
    )

    if (
        static_community_columns
        != temporal_community_columns
    ):

        raise ValueError(
            "Static and temporal community columns are not "
            "aligned in the same order."
        )

    community_columns = (
        static_community_columns
    )

    master_eligible = (
        convert_boolean_column(
            values=master_data[
                "analysis_eligible"
            ],
            column_name=(
                "analysis_eligible"
            ),
        )
    )

    print()
    print(
        f"Master dataset        : "
        f"{MASTER_DATA_FILE}"
    )

    print(
        f"Temporal profiles     : "
        f"{TEMPORAL_PROFILE_FILE}"
    )

    print(
        f"Master rows loaded    : "
        f"{len(master_data)}"
    )

    print(
        f"Eligible master rows  : "
        f"{int(master_eligible.sum())}"
    )

    print(
        f"Companies analysed    : "
        f"{company_profiles['company'].nunique()}"
    )

    print(
        f"Communities analysed  : "
        f"{len(community_columns)}"
    )

    print(
        f"Rolling profiles      : "
        f"{len(temporal_profiles)}"
    )

    print(
        "Metrics               : "
        + ", ".join(
            METRICS
        )
    )

    # ----------------------------------------------------
    # Static sensitivity analysis
    # ----------------------------------------------------

    company_pairs = (
        calculate_company_pair_distances(
            company_profiles=(
                company_profiles
            ),
            community_columns=(
                community_columns
            ),
        )
    )

    company_rankings = (
        build_company_rankings(
            company_pairs
        )
    )

    # ----------------------------------------------------
    # Temporal sensitivity analysis
    # ----------------------------------------------------

    temporal_transitions = (
        calculate_temporal_transitions(
            temporal_profiles=(
                temporal_profiles
            ),
            community_columns=(
                community_columns
            ),
        )
    )

    temporal_rankings = (
        build_temporal_rankings(
            temporal_transitions
        )
    )

    # ----------------------------------------------------
    # Agreement analyses
    # ----------------------------------------------------

    pairwise_agreement = (
        compare_pairwise_distances(
            company_pairs
        )
    )

    static_ranking_agreement = (
        compare_metric_rankings(
            rankings=company_rankings,
            rank_column=(
                "distinctiveness_rank"
            ),
            analysis_type=(
                "company_distinctiveness_rankings"
            ),
        )
    )

    temporal_drift_agreement = (
        compare_metric_rankings(
            rankings=temporal_rankings,
            rank_column=(
                "temporal_drift_rank"
            ),
            analysis_type=(
                "temporal_drift_rankings"
            ),
        )
    )

    temporal_stability_agreement = (
        compare_metric_rankings(
            rankings=temporal_rankings,
            rank_column=(
                "temporal_stability_rank"
            ),
            analysis_type=(
                "temporal_stability_rankings"
            ),
        )
    )

    agreement = pd.concat(
        [
            pairwise_agreement,
            static_ranking_agreement,
            temporal_drift_agreement,
            temporal_stability_agreement,
        ],
        ignore_index=True,
    )

    publication_summary = (
        build_publication_summary(
            agreement=agreement,
            company_pairs=company_pairs,
            company_rankings=company_rankings,
            temporal_transitions=(
                temporal_transitions
            ),
            temporal_rankings=temporal_rankings,
        )
    )

    # ----------------------------------------------------
    # Save publication-quality outputs
    # ----------------------------------------------------

    company_profiles.to_csv(
        COMPANY_PROFILES_FILE,
        index=False,
    )

    company_pairs.to_csv(
        COMPANY_PAIRS_FILE,
        index=False,
    )

    company_rankings.to_csv(
        COMPANY_RANKINGS_FILE,
        index=False,
    )

    temporal_transitions.to_csv(
        TEMPORAL_TRANSITIONS_FILE,
        index=False,
    )

    temporal_rankings.to_csv(
        TEMPORAL_RANKINGS_FILE,
        index=False,
    )

    agreement.to_csv(
        AGREEMENT_FILE,
        index=False,
    )

    publication_summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    # ----------------------------------------------------
    # Additional baseline comparison for console reporting
    # ----------------------------------------------------

    static_baseline_comparison = (
        build_baseline_rank_comparison(
            rankings=company_rankings,
            rank_column=(
                "distinctiveness_rank"
            ),
            value_column=(
                "mean_pairwise_distance"
            ),
            comparison_name=(
                "static_distinctiveness"
            ),
        )
    )

    temporal_baseline_comparison = (
        build_baseline_rank_comparison(
            rankings=temporal_rankings,
            rank_column=(
                "temporal_drift_rank"
            ),
            value_column=(
                "mean_drift"
            ),
            comparison_name=(
                "temporal_drift"
            ),
        )
    )

    # ----------------------------------------------------
    # Console report
    # ----------------------------------------------------

    print()
    print("=" * 70)
    print("STATIC COMPANY DISTINCTIVENESS RANKINGS")
    print("=" * 70)

    static_display_columns = [
        "metric",
        "distinctiveness_rank",
        "company",
        "mean_pairwise_distance",
        "median_pairwise_distance",
        "maximum_pairwise_distance",
    ]

    print(
        company_rankings[
            static_display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("TEMPORAL DRIFT RANKINGS")
    print("=" * 70)

    temporal_display_columns = [
        "metric",
        "temporal_drift_rank",
        "temporal_stability_rank",
        "company",
        "n_transitions",
        "mean_drift",
        "std_drift",
        "maximum_drift",
    ]

    print(
        temporal_rankings[
            temporal_display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("STATIC RANKS RELATIVE TO COSINE")
    print("=" * 70)

    print(
        static_baseline_comparison[
            [
                "metric",
                "company",
                "cosine_rank",
                "metric_rank",
                "absolute_rank_difference",
                "cosine_value",
                "metric_value",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("TEMPORAL RANKS RELATIVE TO COSINE")
    print("=" * 70)

    print(
        temporal_baseline_comparison[
            [
                "metric",
                "company",
                "cosine_rank",
                "metric_rank",
                "absolute_rank_difference",
                "cosine_value",
                "metric_value",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("METRIC AGREEMENT STATISTICS")
    print("=" * 70)

    agreement_display_columns = [
        "analysis_type",
        "metric_a",
        "metric_b",
        "n_observations",
        "spearman_r",
        "kendall_tau",
        "pearson_r",
        "exact_rank_agreement",
        "mean_absolute_rank_difference",
        "maximum_rank_difference",
        "top_rank_agreement",
        "bottom_rank_agreement",
    ]

    print(
        agreement[
            agreement_display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("PUBLICATION SUMMARY")
    print("=" * 70)

    print(
        publication_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("CORRELATION-DISTANCE FALLBACKS")
    print("=" * 70)

    static_fallbacks = int(
        company_pairs[
            "correlation_fallback"
        ].sum()
    )

    temporal_fallbacks = int(
        temporal_transitions[
            "correlation_fallback"
        ].sum()
    )

    print(
        f"Static company pairs : "
        f"{static_fallbacks}"
    )

    print(
        f"Temporal transitions : "
        f"{temporal_fallbacks}"
    )

    print()
    print("=" * 70)
    print("FILES WRITTEN")
    print("=" * 70)

    print(COMPANY_PROFILES_FILE)
    print(COMPANY_PAIRS_FILE)
    print(COMPANY_RANKINGS_FILE)
    print(TEMPORAL_TRANSITIONS_FILE)
    print(TEMPORAL_RANKINGS_FILE)
    print(AGREEMENT_FILE)
    print(SUMMARY_FILE)

    print("=" * 70)


if __name__ == "__main__":
    main()