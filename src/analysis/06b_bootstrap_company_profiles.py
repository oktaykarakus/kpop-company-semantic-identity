"""
WP6b: Bootstrap robustness of company semantic profiles.

This script evaluates sampling uncertainty in the static company
community fingerprints derived from:

    outputs/tables/company_master_analysis_v1.parquet

Songs are resampled with replacement separately within each company.
For every bootstrap replicate, the script reconstructs:

1. Company community fingerprints.
2. Company-level descriptive metrics.
3. Pairwise cosine distances between company fingerprints.
4. Company semantic-distinctiveness rankings.

The script additionally evaluates:

5. The probability of reproducing the complete observed ranking.
6. The probability of preserving the observed top-two and bottom-two sets.
7. The exact-ranking probability relative to a random-order baseline.

This is a reviewer-facing robustness analysis rather than a new
substantive discovery analysis.
"""

from __future__ import annotations

from itertools import combinations
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd


# ======================================================================
# CONFIGURATION
# ======================================================================

ROOT = Path(__file__).resolve().parents[2]

MASTER_PATH = (
    ROOT
    / "outputs"
    / "tables"
    / "company_master_analysis_v1.parquet"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "robustness"
    / "bootstrap_company_profiles"
)

BOOTSTRAP_PROFILES_PATH = (
    OUTPUT_DIR
    / "bootstrap_company_profiles.csv"
)

BOOTSTRAP_PAIRS_PATH = (
    OUTPUT_DIR
    / "bootstrap_company_pairs.csv"
)

BOOTSTRAP_RANKINGS_PATH = (
    OUTPUT_DIR
    / "bootstrap_company_rankings.csv"
)

PROFILE_SUMMARY_PATH = (
    OUTPUT_DIR
    / "bootstrap_company_profile_summary.csv"
)

PAIR_SUMMARY_PATH = (
    OUTPUT_DIR
    / "bootstrap_company_pair_summary.csv"
)

RANKING_SUMMARY_PATH = (
    OUTPUT_DIR
    / "bootstrap_company_ranking_summary.csv"
)

RANKING_STABILITY_PATH = (
    OUTPUT_DIR
    / "bootstrap_company_ranking_stability.csv"
)

PUBLICATION_SUMMARY_PATH = (
    OUTPUT_DIR
    / "bootstrap_company_profiles_summary.csv"
)

N_BOOTSTRAPS = 2000
RANDOM_SEED = 20260717
CONFIDENCE_LEVEL = 0.95
EPSILON = 1e-12

COMPANY_COLUMN = "company"
COMMUNITY_COLUMN = "community"
ELIGIBILITY_COLUMN = "analysis_eligible"

COMPANY_METRIC_COLUMNS = [
    "degree",
    "weighted_degree",
    "bridge_affinity_score",
    "local_novelty",
    "centroid_novelty",
]


# ======================================================================
# VALIDATION AND PREPARATION
# ======================================================================

def validate_master_dataset(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate the authoritative master analysis dataset.
    """

    required_columns = {
        COMPANY_COLUMN,
        COMMUNITY_COLUMN,
        ELIGIBILITY_COLUMN,
        *COMPANY_METRIC_COLUMNS,
    }

    missing_columns = sorted(
        required_columns.difference(
            dataframe.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "The master dataset is missing required columns: "
            + ", ".join(
                missing_columns
            )
        )

    if dataframe.empty:
        raise ValueError(
            "The master dataset is empty."
        )

    if dataframe[
        COMPANY_COLUMN
    ].isna().any():
        raise ValueError(
            "The master dataset contains missing company names."
        )

    if dataframe[
        COMMUNITY_COLUMN
    ].isna().any():
        raise ValueError(
            "The master dataset contains missing community labels."
        )


def prepare_master_dataset(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select analysis-eligible rows and standardise analysis columns.
    """

    working = dataframe.copy()

    eligibility_values = working[
        ELIGIBILITY_COLUMN
    ]

    if pd.api.types.is_bool_dtype(
        eligibility_values
    ):
        eligible_mask = eligibility_values
    else:
        eligible_mask = (
            eligibility_values
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(
                {
                    "true",
                    "1",
                    "yes",
                }
            )
        )

    working = (
        working.loc[
            eligible_mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if working.empty:
        raise ValueError(
            "No analysis-eligible rows were found."
        )

    working[
        COMPANY_COLUMN
    ] = (
        working[
            COMPANY_COLUMN
        ]
        .astype(str)
        .str.strip()
    )

    working[
        COMMUNITY_COLUMN
    ] = (
        pd.to_numeric(
            working[
                COMMUNITY_COLUMN
            ],
            errors="raise",
        )
        .astype(int)
    )

    for column in COMPANY_METRIC_COLUMNS:
        working[
            column
        ] = pd.to_numeric(
            working[
                column
            ],
            errors="coerce",
        )

    return working


def identify_communities(
    dataframe: pd.DataFrame,
) -> list[int]:
    """
    Return the observed community labels in numerical order.
    """

    communities = sorted(
        dataframe[
            COMMUNITY_COLUMN
        ]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    if not communities:
        raise ValueError(
            "No community labels were found."
        )

    return communities


def community_column_name(
    community: int,
) -> str:
    """
    Construct the canonical company-profile column name.
    """

    return f"community_{community}"


# ======================================================================
# DISTANCE
# ======================================================================

def cosine_distance(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
) -> float:
    """
    Calculate cosine distance between two non-zero vectors.
    """

    vector_a = np.asarray(
        vector_a,
        dtype=float,
    )

    vector_b = np.asarray(
        vector_b,
        dtype=float,
    )

    norm_a = float(
        np.linalg.norm(
            vector_a
        )
    )

    norm_b = float(
        np.linalg.norm(
            vector_b
        )
    )

    if (
        norm_a <= EPSILON
        or norm_b <= EPSILON
    ):
        raise ValueError(
            "Cosine distance is undefined for a zero vector."
        )

    similarity = float(
        np.dot(
            vector_a,
            vector_b,
        )
        / (
            norm_a
            * norm_b
        )
    )

    similarity = float(
        np.clip(
            similarity,
            -1.0,
            1.0,
        )
    )

    return float(
        1.0
        - similarity
    )


# ======================================================================
# COMPANY PROFILES
# ======================================================================

def calculate_company_profile(
    company_dataframe: pd.DataFrame,
    company: str,
    communities: list[int],
    bootstrap_id: int,
    is_observed: bool,
) -> dict:
    """
    Construct one company fingerprint and its descriptive metrics.
    """

    n_songs = len(
        company_dataframe
    )

    if n_songs == 0:
        raise ValueError(
            f"Company '{company}' has no rows."
        )

    community_counts = (
        company_dataframe[
            COMMUNITY_COLUMN
        ]
        .value_counts()
    )

    row = {
        "bootstrap_id": int(
            bootstrap_id
        ),
        "is_observed": bool(
            is_observed
        ),
        "company": company,
        "n_songs": int(
            n_songs
        ),
    }

    for community in communities:
        row[
            community_column_name(
                community
            )
        ] = float(
            community_counts.get(
                community,
                0,
            )
            / n_songs
        )

    for metric in COMPANY_METRIC_COLUMNS:
        values = (
            company_dataframe[
                metric
            ]
            .dropna()
            .to_numpy(
                dtype=float
            )
        )

        row[
            f"mean_{metric}"
        ] = (
            float(
                np.mean(
                    values
                )
            )
            if len(
                values
            )
            else np.nan
        )

    fingerprint = np.array(
        [
            row[
                community_column_name(
                    community
                )
            ]
            for community in communities
        ],
        dtype=float,
    )

    positive_probabilities = fingerprint[
        fingerprint > 0
    ]

    entropy = float(
        -np.sum(
            positive_probabilities
            * np.log(
                positive_probabilities
            )
        )
    )

    dominant_index = int(
        np.argmax(
            fingerprint
        )
    )

    row[
        "entropy"
    ] = entropy

    row[
        "effective_communities"
    ] = float(
        np.exp(
            entropy
        )
    )

    row[
        "dominant_community"
    ] = int(
        communities[
            dominant_index
        ]
    )

    row[
        "dominant_community_share"
    ] = float(
        fingerprint[
            dominant_index
        ]
    )

    return row


def calculate_profiles_for_sample(
    sample: pd.DataFrame,
    companies: list[str],
    communities: list[int],
    bootstrap_id: int,
    is_observed: bool,
) -> pd.DataFrame:
    """
    Calculate all company profiles for one sample.
    """

    rows = []

    for company in companies:
        company_dataframe = sample.loc[
            sample[
                COMPANY_COLUMN
            ] == company
        ]

        rows.append(
            calculate_company_profile(
                company_dataframe=company_dataframe,
                company=company,
                communities=communities,
                bootstrap_id=bootstrap_id,
                is_observed=is_observed,
            )
        )

    return pd.DataFrame(
        rows
    )


# ======================================================================
# PAIRWISE DISTANCES AND DISTINCTIVENESS RANKINGS
# ======================================================================

def calculate_company_pairs(
    profiles: pd.DataFrame,
    communities: list[int],
) -> pd.DataFrame:
    """
    Calculate all pairwise cosine distances for one replicate.
    """

    community_columns = [
        community_column_name(
            community
        )
        for community in communities
    ]

    bootstrap_ids = profiles[
        "bootstrap_id"
    ].unique()

    if len(
        bootstrap_ids
    ) != 1:
        raise ValueError(
            "Profiles must belong to exactly one replicate."
        )

    bootstrap_id = int(
        bootstrap_ids[
            0
        ]
    )

    observed_values = profiles[
        "is_observed"
    ].unique()

    if len(
        observed_values
    ) != 1:
        raise ValueError(
            "Profiles contain inconsistent is_observed values."
        )

    is_observed = bool(
        observed_values[
            0
        ]
    )

    profile_lookup = {
        row[
            "company"
        ]: row[
            community_columns
        ].to_numpy(
            dtype=float
        )
        for _, row in profiles.iterrows()
    }

    rows = []

    for company_a, company_b in combinations(
        sorted(
            profile_lookup
        ),
        2,
    ):
        distance = cosine_distance(
            profile_lookup[
                company_a
            ],
            profile_lookup[
                company_b
            ],
        )

        rows.append(
            {
                "bootstrap_id": bootstrap_id,
                "is_observed": is_observed,
                "company_a": company_a,
                "company_b": company_b,
                "company_pair": (
                    f"{company_a}|{company_b}"
                ),
                "cosine_distance": distance,
            }
        )

    return pd.DataFrame(
        rows
    )


def calculate_company_rankings(
    pairs: pd.DataFrame,
    companies: list[str],
) -> pd.DataFrame:
    """
    Rank companies by mean pairwise cosine distance.
    """

    bootstrap_ids = pairs[
        "bootstrap_id"
    ].unique()

    if len(
        bootstrap_ids
    ) != 1:
        raise ValueError(
            "Pairwise rows must belong to exactly one replicate."
        )

    bootstrap_id = int(
        bootstrap_ids[
            0
        ]
    )

    observed_values = pairs[
        "is_observed"
    ].unique()

    if len(
        observed_values
    ) != 1:
        raise ValueError(
            "Pairwise rows contain inconsistent is_observed values."
        )

    is_observed = bool(
        observed_values[
            0
        ]
    )

    rows = []

    for company in companies:
        company_distances = pairs.loc[
            (
                pairs[
                    "company_a"
                ] == company
            )
            |
            (
                pairs[
                    "company_b"
                ] == company
            ),
            "cosine_distance",
        ].to_numpy(
            dtype=float
        )

        expected_distances = (
            len(
                companies
            )
            - 1
        )

        if len(
            company_distances
        ) != expected_distances:
            raise ValueError(
                "Unexpected number of pairwise distances for "
                f"company '{company}'."
            )

        rows.append(
            {
                "bootstrap_id": bootstrap_id,
                "is_observed": is_observed,
                "company": company,
                "mean_pairwise_distance": float(
                    np.mean(
                        company_distances
                    )
                ),
                "median_pairwise_distance": float(
                    np.median(
                        company_distances
                    )
                ),
                "maximum_pairwise_distance": float(
                    np.max(
                        company_distances
                    )
                ),
            }
        )

    rankings = pd.DataFrame(
        rows
    )

    rankings[
        "distinctiveness_rank"
    ] = (
        rankings[
            "mean_pairwise_distance"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    rankings = (
        rankings
        .sort_values(
            [
                "distinctiveness_rank",
                "company",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return rankings


# ======================================================================
# BOOTSTRAP
# ======================================================================

def bootstrap_company_sample(
    master: pd.DataFrame,
    companies: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Resample songs with replacement separately within each company.

    Each company retains its observed sample size.
    """

    sampled_frames = []

    for company in companies:
        company_dataframe = (
            master.loc[
                master[
                    COMPANY_COLUMN
                ] == company
            ]
            .reset_index(
                drop=True
            )
        )

        n_company_rows = len(
            company_dataframe
        )

        if n_company_rows == 0:
            raise ValueError(
                f"Company '{company}' has no eligible rows."
            )

        sampled_indices = rng.integers(
            low=0,
            high=n_company_rows,
            size=n_company_rows,
        )

        sampled_company = (
            company_dataframe.iloc[
                sampled_indices
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        sampled_frames.append(
            sampled_company
        )

    return pd.concat(
        sampled_frames,
        ignore_index=True,
    )


def run_bootstrap(
    master: pd.DataFrame,
    companies: list[str],
    communities: list[int],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Run the observed analysis and all bootstrap replicates.
    """

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    profile_frames = []
    pair_frames = []
    ranking_frames = []

    observed_profiles = (
        calculate_profiles_for_sample(
            sample=master,
            companies=companies,
            communities=communities,
            bootstrap_id=0,
            is_observed=True,
        )
    )

    observed_pairs = (
        calculate_company_pairs(
            profiles=observed_profiles,
            communities=communities,
        )
    )

    observed_rankings = (
        calculate_company_rankings(
            pairs=observed_pairs,
            companies=companies,
        )
    )

    profile_frames.append(
        observed_profiles
    )

    pair_frames.append(
        observed_pairs
    )

    ranking_frames.append(
        observed_rankings
    )

    progress_interval = max(
        1,
        N_BOOTSTRAPS
        // 20,
    )

    for bootstrap_id in range(
        1,
        N_BOOTSTRAPS + 1,
    ):
        bootstrap_sample = (
            bootstrap_company_sample(
                master=master,
                companies=companies,
                rng=rng,
            )
        )

        bootstrap_profiles = (
            calculate_profiles_for_sample(
                sample=bootstrap_sample,
                companies=companies,
                communities=communities,
                bootstrap_id=bootstrap_id,
                is_observed=False,
            )
        )

        bootstrap_pairs = (
            calculate_company_pairs(
                profiles=bootstrap_profiles,
                communities=communities,
            )
        )

        bootstrap_rankings = (
            calculate_company_rankings(
                pairs=bootstrap_pairs,
                companies=companies,
            )
        )

        profile_frames.append(
            bootstrap_profiles
        )

        pair_frames.append(
            bootstrap_pairs
        )

        ranking_frames.append(
            bootstrap_rankings
        )

        if (
            bootstrap_id
            % progress_interval
            == 0
            or bootstrap_id
            == N_BOOTSTRAPS
        ):
            print(
                "Bootstrap progress     : "
                f"{bootstrap_id}/{N_BOOTSTRAPS}"
            )

    profiles = pd.concat(
        profile_frames,
        ignore_index=True,
    )

    pairs = pd.concat(
        pair_frames,
        ignore_index=True,
    )

    rankings = pd.concat(
        ranking_frames,
        ignore_index=True,
    )

    return (
        profiles,
        pairs,
        rankings,
    )


# ======================================================================
# CONFIDENCE INTERVALS
# ======================================================================

def confidence_limits() -> tuple[
    float,
    float,
]:
    """
    Return percentile limits for the configured confidence level.
    """

    alpha = (
        1.0
        - CONFIDENCE_LEVEL
    )

    return (
        alpha
        / 2.0,
        1.0
        - alpha
        / 2.0,
    )


def summarise_numeric_bootstrap(
    bootstrap_values: pd.Series,
    observed_value: float,
) -> dict:
    """
    Calculate bootstrap mean, uncertainty, bias and percentile interval.
    """

    values = (
        pd.to_numeric(
            bootstrap_values,
            errors="coerce",
        )
        .dropna()
        .to_numpy(
            dtype=float
        )
    )

    if len(
        values
    ) == 0:
        return {
            "observed_value": observed_value,
            "bootstrap_mean": np.nan,
            "bootstrap_median": np.nan,
            "bootstrap_std": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "absolute_ci_width": np.nan,
            "bootstrap_bias": np.nan,
            "relative_ci_width": np.nan,
        }

    lower_quantile, upper_quantile = (
        confidence_limits()
    )

    ci_lower = float(
        np.quantile(
            values,
            lower_quantile,
        )
    )

    ci_upper = float(
        np.quantile(
            values,
            upper_quantile,
        )
    )

    bootstrap_mean = float(
        np.mean(
            values
        )
    )

    absolute_ci_width = float(
        ci_upper
        - ci_lower
    )

    absolute_observed = abs(
        observed_value
    )

    relative_ci_width = (
        float(
            absolute_ci_width
            / absolute_observed
        )
        if absolute_observed > EPSILON
        else np.nan
    )

    return {
        "observed_value": float(
            observed_value
        ),
        "bootstrap_mean": bootstrap_mean,
        "bootstrap_median": float(
            np.median(
                values
            )
        ),
        "bootstrap_std": float(
            np.std(
                values,
                ddof=1,
            )
        ),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "absolute_ci_width": absolute_ci_width,
        "bootstrap_bias": float(
            bootstrap_mean
            - observed_value
        ),
        "relative_ci_width": relative_ci_width,
    }


# ======================================================================
# PROFILE SUMMARY
# ======================================================================

def build_profile_summary(
    profiles: pd.DataFrame,
    communities: list[int],
) -> pd.DataFrame:
    """
    Summarise uncertainty in company profile components and metrics.
    """

    observed = profiles.loc[
        profiles[
            "is_observed"
        ]
    ]

    bootstrap = profiles.loc[
        ~profiles[
            "is_observed"
        ]
    ]

    profile_value_columns = [
        community_column_name(
            community
        )
        for community in communities
    ]

    profile_value_columns.extend(
        [
            f"mean_{metric}"
            for metric in COMPANY_METRIC_COLUMNS
        ]
    )

    profile_value_columns.extend(
        [
            "entropy",
            "effective_communities",
            "dominant_community_share",
        ]
    )

    rows = []

    for company in sorted(
        profiles[
            "company"
        ].unique()
    ):
        observed_company = observed.loc[
            observed[
                "company"
            ] == company
        ]

        bootstrap_company = bootstrap.loc[
            bootstrap[
                "company"
            ] == company
        ]

        if len(
            observed_company
        ) != 1:
            raise ValueError(
                "Expected one observed company profile for "
                f"'{company}'."
            )

        observed_row = observed_company.iloc[
            0
        ]

        for variable in profile_value_columns:
            summary = summarise_numeric_bootstrap(
                bootstrap_values=bootstrap_company[
                    variable
                ],
                observed_value=float(
                    observed_row[
                        variable
                    ]
                ),
            )

            rows.append(
                {
                    "company": company,
                    "variable": variable,
                    **summary,
                    "observed_within_ci": bool(
                        summary[
                            "ci_lower"
                        ]
                        <= summary[
                            "observed_value"
                        ]
                        <= summary[
                            "ci_upper"
                        ]
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ======================================================================
# PAIR SUMMARY
# ======================================================================

def build_pair_summary(
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarise bootstrap uncertainty for every company pair.
    """

    observed = pairs.loc[
        pairs[
            "is_observed"
        ]
    ]

    bootstrap = pairs.loc[
        ~pairs[
            "is_observed"
        ]
    ]

    rows = []

    for company_pair in sorted(
        pairs[
            "company_pair"
        ].unique()
    ):
        observed_pair = observed.loc[
            observed[
                "company_pair"
            ] == company_pair
        ]

        bootstrap_pair = bootstrap.loc[
            bootstrap[
                "company_pair"
            ] == company_pair
        ]

        if len(
            observed_pair
        ) != 1:
            raise ValueError(
                "Expected one observed row for company pair "
                f"'{company_pair}'."
            )

        observed_row = observed_pair.iloc[
            0
        ]

        summary = summarise_numeric_bootstrap(
            bootstrap_values=bootstrap_pair[
                "cosine_distance"
            ],
            observed_value=float(
                observed_row[
                    "cosine_distance"
                ]
            ),
        )

        rows.append(
            {
                "company_a": observed_row[
                    "company_a"
                ],
                "company_b": observed_row[
                    "company_b"
                ],
                "company_pair": company_pair,
                **summary,
                "observed_within_ci": bool(
                    summary[
                        "ci_lower"
                    ]
                    <= summary[
                        "observed_value"
                    ]
                    <= summary[
                        "ci_upper"
                    ]
                ),
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "observed_value",
                "company_pair",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )


# ======================================================================
# COMPANY RANKING SUMMARY
# ======================================================================

def build_ranking_summary(
    rankings: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarise company ranking stability across bootstrap replicates.
    """

    observed = rankings.loc[
        rankings[
            "is_observed"
        ]
    ]

    bootstrap = rankings.loc[
        ~rankings[
            "is_observed"
        ]
    ]

    companies = sorted(
        rankings[
            "company"
        ].unique()
    )

    n_companies = len(
        companies
    )

    rows = []

    for company in companies:
        observed_company = observed.loc[
            observed[
                "company"
            ] == company
        ]

        bootstrap_company = bootstrap.loc[
            bootstrap[
                "company"
            ] == company
        ]

        if len(
            observed_company
        ) != 1:
            raise ValueError(
                "Expected one observed ranking for company "
                f"'{company}'."
            )

        observed_row = observed_company.iloc[
            0
        ]

        distance_summary = (
            summarise_numeric_bootstrap(
                bootstrap_values=bootstrap_company[
                    "mean_pairwise_distance"
                ],
                observed_value=float(
                    observed_row[
                        "mean_pairwise_distance"
                    ]
                ),
            )
        )

        rank_values = (
            bootstrap_company[
                "distinctiveness_rank"
            ]
            .astype(int)
        )

        row = {
            "company": company,
            "observed_rank": int(
                observed_row[
                    "distinctiveness_rank"
                ]
            ),
            "observed_mean_pairwise_distance": float(
                observed_row[
                    "mean_pairwise_distance"
                ]
            ),
            "bootstrap_mean_pairwise_distance": (
                distance_summary[
                    "bootstrap_mean"
                ]
            ),
            "distance_ci_lower": (
                distance_summary[
                    "ci_lower"
                ]
            ),
            "distance_ci_upper": (
                distance_summary[
                    "ci_upper"
                ]
            ),
            "distance_bootstrap_std": (
                distance_summary[
                    "bootstrap_std"
                ]
            ),
            "distance_bootstrap_bias": (
                distance_summary[
                    "bootstrap_bias"
                ]
            ),
            "mean_bootstrap_rank": float(
                rank_values.mean()
            ),
            "median_bootstrap_rank": float(
                rank_values.median()
            ),
            "rank_std": float(
                rank_values.std(
                    ddof=1
                )
            ),
            "observed_rank_probability": float(
                (
                    rank_values
                    == int(
                        observed_row[
                            "distinctiveness_rank"
                        ]
                    )
                ).mean()
            ),
            "top_rank_probability": float(
                (
                    rank_values
                    == 1
                ).mean()
            ),
            "top_two_probability": float(
                (
                    rank_values
                    <= 2
                ).mean()
            ),
            "bottom_rank_probability": float(
                (
                    rank_values
                    == n_companies
                ).mean()
            ),
            "bottom_two_probability": float(
                (
                    rank_values
                    >= (
                        n_companies
                        - 1
                    )
                ).mean()
            ),
        }

        for rank in range(
            1,
            n_companies + 1,
        ):
            row[
                f"rank_{rank}_probability"
            ] = float(
                (
                    rank_values
                    == rank
                ).mean()
            )

        rows.append(
            row
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "observed_rank",
                "company",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ======================================================================
# GLOBAL RANKING STABILITY
# ======================================================================

def calculate_complete_ranking_stability(
    rankings: pd.DataFrame,
) -> dict:
    """
    Evaluate stability of the complete company ranking.

    The function calculates:

    - exact reproduction of the complete observed order;
    - preservation of the observed top-two company set;
    - preservation of the observed bottom-two company set;
    - preservation of both sets simultaneously;
    - the exact-order probability expected under random ordering;
    - the ratio between observed and random exact-order probabilities.
    """

    observed = (
        rankings.loc[
            rankings[
                "is_observed"
            ]
        ]
        .sort_values(
            [
                "distinctiveness_rank",
                "company",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    bootstrap = rankings.loc[
        ~rankings[
            "is_observed"
        ]
    ]

    n_companies = len(
        observed
    )

    if n_companies < 2:
        raise ValueError(
            "At least two companies are required for ranking stability."
        )

    observed_order = tuple(
        observed[
            "company"
        ].tolist()
    )

    observed_top_two = frozenset(
        observed.head(
            2
        )[
            "company"
        ].tolist()
    )

    observed_bottom_two = frozenset(
        observed.tail(
            2
        )[
            "company"
        ].tolist()
    )

    exact_matches = 0
    top_two_matches = 0
    bottom_two_matches = 0
    joint_group_matches = 0

    bootstrap_ids = sorted(
        bootstrap[
            "bootstrap_id"
        ]
        .unique()
        .tolist()
    )

    if not bootstrap_ids:
        raise ValueError(
            "No bootstrap ranking replicates were found."
        )

    for bootstrap_id in bootstrap_ids:
        replicate = (
            bootstrap.loc[
                bootstrap[
                    "bootstrap_id"
                ] == bootstrap_id
            ]
            .sort_values(
                [
                    "distinctiveness_rank",
                    "company",
                ]
            )
            .reset_index(
                drop=True
            )
        )

        if len(
            replicate
        ) != n_companies:
            raise ValueError(
                "Bootstrap replicate "
                f"{bootstrap_id} contains an unexpected number "
                "of company rankings."
            )

        replicate_order = tuple(
            replicate[
                "company"
            ].tolist()
        )

        replicate_top_two = frozenset(
            replicate.head(
                2
            )[
                "company"
            ].tolist()
        )

        replicate_bottom_two = frozenset(
            replicate.tail(
                2
            )[
                "company"
            ].tolist()
        )

        exact_match = (
            replicate_order
            == observed_order
        )

        top_match = (
            replicate_top_two
            == observed_top_two
        )

        bottom_match = (
            replicate_bottom_two
            == observed_bottom_two
        )

        exact_matches += int(
            exact_match
        )

        top_two_matches += int(
            top_match
        )

        bottom_two_matches += int(
            bottom_match
        )

        joint_group_matches += int(
            top_match
            and bottom_match
        )

    n_bootstrap_replicates = len(
        bootstrap_ids
    )

    exact_order_probability = float(
        exact_matches
        / n_bootstrap_replicates
    )

    random_exact_order_probability = float(
        1.0
        / factorial(
            n_companies
        )
    )

    exact_order_stability_ratio = float(
        exact_order_probability
        / random_exact_order_probability
    )

    return {
        "n_companies": int(
            n_companies
        ),
        "n_bootstrap_replicates": int(
            n_bootstrap_replicates
        ),
        "observed_order": "|".join(
            observed_order
        ),
        "observed_top_two": "|".join(
            sorted(
                observed_top_two
            )
        ),
        "observed_bottom_two": "|".join(
            sorted(
                observed_bottom_two
            )
        ),
        "exact_order_probability": (
            exact_order_probability
        ),
        "random_exact_order_probability": (
            random_exact_order_probability
        ),
        "exact_order_stability_ratio": (
            exact_order_stability_ratio
        ),
        "top_two_set_probability": float(
            top_two_matches
            / n_bootstrap_replicates
        ),
        "bottom_two_set_probability": float(
            bottom_two_matches
            / n_bootstrap_replicates
        ),
        "joint_group_probability": float(
            joint_group_matches
            / n_bootstrap_replicates
        ),
    }


# ======================================================================
# ROBUSTNESS CLASSIFICATIONS
# ======================================================================

def classify_rank_stability(
    probability: float,
) -> str:
    """
    Classify the probability of reproducing an observed rank.
    """

    if probability >= 0.80:
        return "highly_stable"

    if probability >= 0.60:
        return "stable"

    if probability >= 0.40:
        return "moderately_stable"

    return "rank_uncertain"


def classify_group_stability(
    probability: float,
) -> str:
    """
    Classify preservation of an observed company group.
    """

    if probability >= 0.95:
        return "highly_robust"

    if probability >= 0.80:
        return "robust"

    if probability >= 0.60:
        return "moderately_robust"

    return "sampling_sensitive"


def classify_global_exact_order(
    exact_probability: float,
    stability_ratio: float,
) -> str:
    """
    Classify complete-ranking stability relative to random ordering.

    Exact-order probabilities are naturally lower than individual-rank
    probabilities because every company must occupy the correct position.
    The random-order comparison is therefore also considered.
    """

    if (
        exact_probability >= 0.50
        and stability_ratio >= 5.0
    ):
        return "strong_complete_order_stability"

    if (
        exact_probability >= 0.25
        and stability_ratio >= 4.0
    ):
        return "moderate_complete_order_stability"

    if stability_ratio >= 2.0:
        return "above_random_order_stability"

    return "weak_complete_order_stability"


def classify_pair_uncertainty(
    bootstrap_std: float,
    observed_within_ci: bool,
) -> str:
    """
    Classify pairwise uncertainty without dividing by a near-zero distance.
    """

    if not observed_within_ci:
        return "bootstrap_inconsistent"

    if bootstrap_std <= 0.025:
        return "comparatively_precise"

    if bootstrap_std <= 0.050:
        return "moderate_sampling_uncertainty"

    return "substantial_sampling_uncertainty"


# ======================================================================
# PUBLICATION SUMMARY
# ======================================================================

def build_publication_summary(
    ranking_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    ranking_stability_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct a compact reviewer-facing bootstrap summary.

    This function creates:

    - one global complete-ranking row;
    - one row for each company's observed ranking;
    - one row for each unique company pair.

    A new dictionary is appended for every row to avoid accidental reuse
    of values across iterations.
    """

    if len(
        ranking_stability_summary
    ) != 1:
        raise ValueError(
            "Expected exactly one global ranking-stability row."
        )

    rows = []

    global_stability = (
        ranking_stability_summary.iloc[
            0
        ]
    )

    exact_order_probability = float(
        global_stability[
            "exact_order_probability"
        ]
    )

    random_order_probability = float(
        global_stability[
            "random_exact_order_probability"
        ]
    )

    exact_order_stability_ratio = float(
        global_stability[
            "exact_order_stability_ratio"
        ]
    )

    joint_group_probability = float(
        global_stability[
            "joint_group_probability"
        ]
    )

    # ------------------------------------------------------------------
    # Global ranking row
    # ------------------------------------------------------------------

    rows.append(
        {
            "analysis": (
                "Global distinctiveness ranking"
            ),
            "entity": (
                "All companies"
            ),
            "finding_tested": (
                "Observed complete company ordering"
            ),
            "observed_value": np.nan,
            "bootstrap_probability": (
                exact_order_probability
            ),
            "random_probability": (
                random_order_probability
            ),
            "stability_ratio": (
                exact_order_stability_ratio
            ),
            "secondary_test": (
                "Observed top-two and bottom-two sets"
            ),
            "secondary_probability": (
                joint_group_probability
            ),
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "absolute_ci_width": np.nan,
            "bootstrap_bias": np.nan,
            "conclusion": (
                classify_global_exact_order(
                    exact_probability=(
                        exact_order_probability
                    ),
                    stability_ratio=(
                        exact_order_stability_ratio
                    ),
                )
            ),
            "group_conclusion": (
                classify_group_stability(
                    joint_group_probability
                )
            ),
        }
    )

    # ------------------------------------------------------------------
    # Company-specific ranking rows
    # ------------------------------------------------------------------

    for _, ranking_row in ranking_summary.iterrows():
        observed_rank = int(
            ranking_row[
                "observed_rank"
            ]
        )

        if observed_rank <= 2:
            group_probability = float(
                ranking_row[
                    "top_two_probability"
                ]
            )

            group_tested = (
                "Top-two distinctiveness membership"
            )
        else:
            group_probability = float(
                ranking_row[
                    "bottom_two_probability"
                ]
            )

            group_tested = (
                "Bottom-two distinctiveness membership"
            )

        rows.append(
            {
                "analysis": (
                    "Company distinctiveness rank"
                ),
                "entity": (
                    ranking_row[
                        "company"
                    ]
                ),
                "finding_tested": (
                    f"Observed distinctiveness rank "
                    f"{observed_rank}"
                ),
                "observed_value": float(
                    observed_rank
                ),
                "bootstrap_probability": float(
                    ranking_row[
                        "observed_rank_probability"
                    ]
                ),
                "random_probability": float(
                    1.0
                    / len(
                        ranking_summary
                    )
                ),
                "stability_ratio": float(
                    ranking_row[
                        "observed_rank_probability"
                    ]
                    / (
                        1.0
                        / len(
                            ranking_summary
                        )
                    )
                ),
                "secondary_test": (
                    group_tested
                ),
                "secondary_probability": (
                    group_probability
                ),
                "ci_lower": float(
                    ranking_row[
                        "distance_ci_lower"
                    ]
                ),
                "ci_upper": float(
                    ranking_row[
                        "distance_ci_upper"
                    ]
                ),
                "absolute_ci_width": float(
                    ranking_row[
                        "distance_ci_upper"
                    ]
                    - ranking_row[
                        "distance_ci_lower"
                    ]
                ),
                "bootstrap_bias": float(
                    ranking_row[
                        "distance_bootstrap_bias"
                    ]
                ),
                "conclusion": (
                    classify_rank_stability(
                        float(
                            ranking_row[
                                "observed_rank_probability"
                            ]
                        )
                    )
                ),
                "group_conclusion": (
                    classify_group_stability(
                        group_probability
                    )
                ),
            }
        )

    # ------------------------------------------------------------------
    # Pairwise-distance rows
    # ------------------------------------------------------------------

    for _, pair_row in pair_summary.iterrows():
        observed_within_ci = bool(
            pair_row[
                "observed_within_ci"
            ]
        )

        bootstrap_std = float(
            pair_row[
                "bootstrap_std"
            ]
        )

        rows.append(
            {
                "analysis": (
                    "Pairwise company distance"
                ),
                "entity": (
                    pair_row[
                        "company_pair"
                    ]
                ),
                "finding_tested": (
                    "Observed cosine distance"
                ),
                "observed_value": float(
                    pair_row[
                        "observed_value"
                    ]
                ),
                "bootstrap_probability": np.nan,
                "random_probability": np.nan,
                "stability_ratio": np.nan,
                "secondary_test": (
                    "Observed estimate within percentile interval"
                ),
                "secondary_probability": float(
                    observed_within_ci
                ),
                "ci_lower": float(
                    pair_row[
                        "ci_lower"
                    ]
                ),
                "ci_upper": float(
                    pair_row[
                        "ci_upper"
                    ]
                ),
                "absolute_ci_width": float(
                    pair_row[
                        "absolute_ci_width"
                    ]
                ),
                "bootstrap_bias": float(
                    pair_row[
                        "bootstrap_bias"
                    ]
                ),
                "conclusion": (
                    classify_pair_uncertainty(
                        bootstrap_std=bootstrap_std,
                        observed_within_ci=(
                            observed_within_ci
                        ),
                    )
                ),
                "group_conclusion": "",
            }
        )

    publication_summary = pd.DataFrame(
        rows
    )

    expected_rows = (
        1
        + len(
            ranking_summary
        )
        + len(
            pair_summary
        )
    )

    if len(
        publication_summary
    ) != expected_rows:
        raise RuntimeError(
            "Publication summary contains an unexpected number "
            f"of rows: expected {expected_rows}, "
            f"received {len(publication_summary)}."
        )

    expected_pairs = set(
        pair_summary[
            "company_pair"
        ].tolist()
    )

    publication_pairs = set(
        publication_summary.loc[
            publication_summary[
                "analysis"
            ] == "Pairwise company distance",
            "entity",
        ].tolist()
    )

    if publication_pairs != expected_pairs:
        raise RuntimeError(
            "Publication summary pairwise entities do not match "
            "the pair-summary table."
        )

    return publication_summary


# ======================================================================
# CONSOLE OUTPUT
# ======================================================================

def print_heading(
    title: str,
) -> None:
    """
    Print a consistent console heading.
    """

    print()
    print(
        "="
        * 70
    )
    print(
        title
    )
    print(
        "="
        * 70
    )


def print_results(
    master: pd.DataFrame,
    companies: list[str],
    communities: list[int],
    ranking_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    ranking_stability_summary: pd.DataFrame,
    publication_summary: pd.DataFrame,
) -> None:
    """
    Print the principal bootstrap results.
    """

    print_heading(
        "WP6 : BOOTSTRAP COMPANY PROFILES"
    )

    print()
    print(
        f"Master dataset        : {MASTER_PATH}"
    )
    print(
        f"Eligible master rows  : {len(master)}"
    )
    print(
        f"Companies analysed    : {len(companies)}"
    )
    print(
        f"Communities analysed  : {len(communities)}"
    )
    print(
        f"Bootstrap replicates  : {N_BOOTSTRAPS}"
    )
    print(
        f"Random seed           : {RANDOM_SEED}"
    )
    print(
        "Confidence level      : "
        f"{CONFIDENCE_LEVEL:.1%}"
    )

    print_heading(
        "BOOTSTRAP DISTINCTIVENESS RANKING SUMMARY"
    )

    ranking_columns = [
        "company",
        "observed_rank",
        "observed_mean_pairwise_distance",
        "bootstrap_mean_pairwise_distance",
        "distance_ci_lower",
        "distance_ci_upper",
        "observed_rank_probability",
        "top_rank_probability",
        "top_two_probability",
        "bottom_two_probability",
    ]

    print(
        ranking_summary[
            ranking_columns
        ].to_string(
            index=False
        )
    )

    print_heading(
        "BOOTSTRAP PAIRWISE DISTANCE SUMMARY"
    )

    pair_columns = [
        "company_pair",
        "observed_value",
        "bootstrap_mean",
        "bootstrap_std",
        "ci_lower",
        "ci_upper",
        "absolute_ci_width",
        "bootstrap_bias",
    ]

    print(
        pair_summary[
            pair_columns
        ].to_string(
            index=False
        )
    )

    print_heading(
        "GLOBAL RANKING STABILITY"
    )

    global_row = (
        ranking_stability_summary.iloc[
            0
        ]
    )

    print(
        "Observed order           : "
        f"{global_row['observed_order']}"
    )

    print(
        "Exact-order probability  : "
        f"{global_row['exact_order_probability']:.4f}"
    )

    print(
        "Random-order probability : "
        f"{global_row['random_exact_order_probability']:.4f}"
    )

    print(
        "Order stability ratio    : "
        f"{global_row['exact_order_stability_ratio']:.4f}"
    )

    print(
        "Top-two-set probability  : "
        f"{global_row['top_two_set_probability']:.4f}"
    )

    print(
        "Bottom-two probability   : "
        f"{global_row['bottom_two_set_probability']:.4f}"
    )

    print(
        "Joint grouping probability: "
        f"{global_row['joint_group_probability']:.4f}"
    )

    print_heading(
        "PUBLICATION SUMMARY"
    )

    print(
        publication_summary.to_string(
            index=False
        )
    )


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    """
    Execute WP6b bootstrap robustness analysis.
    """

    print_heading(
        "WP6 : BOOTSTRAP COMPANY PROFILES"
    )

    if not MASTER_PATH.exists():
        raise FileNotFoundError(
            "The authoritative master dataset does not exist: "
            f"{MASTER_PATH}"
        )

    master_raw = pd.read_parquet(
        MASTER_PATH
    )

    validate_master_dataset(
        master_raw
    )

    master = prepare_master_dataset(
        master_raw
    )

    companies = sorted(
        master[
            COMPANY_COLUMN
        ]
        .unique()
        .tolist()
    )

    communities = identify_communities(
        master
    )

    if len(
        companies
    ) < 2:
        raise ValueError(
            "At least two companies are required."
        )

    print()
    print(
        f"Master dataset        : {MASTER_PATH}"
    )
    print(
        f"Master rows loaded    : {len(master_raw)}"
    )
    print(
        f"Eligible master rows  : {len(master)}"
    )
    print(
        f"Companies analysed    : {len(companies)}"
    )
    print(
        f"Communities analysed  : {len(communities)}"
    )
    print(
        f"Bootstrap replicates  : {N_BOOTSTRAPS}"
    )

    (
        profiles,
        pairs,
        rankings,
    ) = run_bootstrap(
        master=master,
        companies=companies,
        communities=communities,
    )

    profile_summary = (
        build_profile_summary(
            profiles=profiles,
            communities=communities,
        )
    )

    pair_summary = (
        build_pair_summary(
            pairs=pairs
        )
    )

    ranking_summary = (
        build_ranking_summary(
            rankings=rankings
        )
    )

    ranking_stability = (
        calculate_complete_ranking_stability(
            rankings=rankings
        )
    )

    ranking_stability_summary = (
        pd.DataFrame(
            [
                ranking_stability
            ]
        )
    )

    publication_summary = (
        build_publication_summary(
            ranking_summary=ranking_summary,
            pair_summary=pair_summary,
            ranking_stability_summary=(
                ranking_stability_summary
            ),
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    profiles.to_csv(
        BOOTSTRAP_PROFILES_PATH,
        index=False,
    )

    pairs.to_csv(
        BOOTSTRAP_PAIRS_PATH,
        index=False,
    )

    rankings.to_csv(
        BOOTSTRAP_RANKINGS_PATH,
        index=False,
    )

    profile_summary.to_csv(
        PROFILE_SUMMARY_PATH,
        index=False,
    )

    pair_summary.to_csv(
        PAIR_SUMMARY_PATH,
        index=False,
    )

    ranking_summary.to_csv(
        RANKING_SUMMARY_PATH,
        index=False,
    )

    ranking_stability_summary.to_csv(
        RANKING_STABILITY_PATH,
        index=False,
    )

    publication_summary.to_csv(
        PUBLICATION_SUMMARY_PATH,
        index=False,
    )

    print_results(
        master=master,
        companies=companies,
        communities=communities,
        ranking_summary=ranking_summary,
        pair_summary=pair_summary,
        ranking_stability_summary=(
            ranking_stability_summary
        ),
        publication_summary=publication_summary,
    )

    print_heading(
        "FILES WRITTEN"
    )

    for output_path in [
        BOOTSTRAP_PROFILES_PATH,
        BOOTSTRAP_PAIRS_PATH,
        BOOTSTRAP_RANKINGS_PATH,
        PROFILE_SUMMARY_PATH,
        PAIR_SUMMARY_PATH,
        RANKING_SUMMARY_PATH,
        RANKING_STABILITY_PATH,
        PUBLICATION_SUMMARY_PATH,
    ]:
        print(
            output_path
        )

    print(
        "="
        * 70
    )


if __name__ == "__main__":
    main()