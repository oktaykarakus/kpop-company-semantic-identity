"""
03_company_semantic_profiles.py

Constructs company-level semantic profiles across the validated lyric
communities.

Main outputs
------------
1. Company-by-community song counts
2. Company-normalised community shares
3. Community-normalised company shares
4. Company diversity statistics
5. Community concentration and dominance statistics
6. Pairwise similarity between company semantic fingerprints
7. Publication-ready summary tables

The analysis uses the frozen company master dataset created by:
01_build_master_dataset.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


# =============================================================================
# Paths
# =============================================================================

COMPANY_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    COMPANY_ROOT
    / "outputs"
    / "tables"
    / "company_master_analysis_v1.parquet"
)

OUTPUT_ROOT = (
    COMPANY_ROOT
    / "outputs"
    / "company_profiles"
)

RAW_OUTPUT_DIR = OUTPUT_ROOT / "raw"
TABLE_OUTPUT_DIR = OUTPUT_ROOT / "tables"
REPORT_OUTPUT_DIR = OUTPUT_ROOT / "reports"

for directory in [
    RAW_OUTPUT_DIR,
    TABLE_OUTPUT_DIR,
    REPORT_OUTPUT_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Configuration
# =============================================================================

COMPANY_ORDER = ["HYBE", "JYP", "SM", "YG"]

REQUIRED_COLUMNS = {
    "song_uid",
    "title",
    "artist",
    "artist_normalised",
    "company",
    "year",
    "community",
    "participation_coefficient",
    "community_entropy",
    "external_strength_fraction",
    "external_communities_reached",
    "centroid_novelty",
    "local_novelty",
}

EPSILON = 1e-12


# =============================================================================
# Validation
# =============================================================================

def validate_input(df: pd.DataFrame) -> None:
    """Validate the frozen company analysis dataset."""
    missing = REQUIRED_COLUMNS.difference(df.columns)

    if missing:
        raise ValueError(
            "Input dataset is missing required columns:\n"
            + "\n".join(f"  - {column}" for column in sorted(missing))
        )

    if df.empty:
        raise ValueError("Input dataset is empty.")

    if df["song_uid"].duplicated().any():
        n_duplicates = int(df["song_uid"].duplicated().sum())
        raise ValueError(
            f"Input contains {n_duplicates:,} duplicated song identifiers."
        )

    invalid_companies = sorted(
        set(df["company"].dropna().astype(str)).difference(COMPANY_ORDER)
    )

    if invalid_companies:
        raise ValueError(
            f"Unexpected company labels found: {invalid_companies}"
        )

    if df["company"].isna().any():
        raise ValueError("Missing company labels are present.")

    if df["community"].isna().any():
        raise ValueError("Missing community assignments are present.")


# =============================================================================
# Diversity functions
# =============================================================================

def shannon_entropy(probabilities: np.ndarray) -> float:
    """Calculate Shannon entropy using natural logarithms."""
    probabilities = probabilities[probabilities > 0]

    if probabilities.size == 0:
        return np.nan

    return float(
        -np.sum(probabilities * np.log(probabilities))
    )


def normalised_shannon_entropy(
    probabilities: np.ndarray,
    total_possible_categories: int,
) -> float:
    """
    Shannon entropy divided by the maximum possible entropy.

    This produces a value between 0 and 1.
    """
    if total_possible_categories <= 1:
        return np.nan

    entropy = shannon_entropy(probabilities)
    maximum_entropy = np.log(total_possible_categories)

    return float(entropy / maximum_entropy)


def effective_number_of_communities(
    probabilities: np.ndarray,
) -> float:
    """
    Convert Shannon entropy into the effective number of equally represented
    communities.
    """
    entropy = shannon_entropy(probabilities)

    if np.isnan(entropy):
        return np.nan

    return float(np.exp(entropy))


def simpson_diversity(probabilities: np.ndarray) -> float:
    """
    Simpson diversity expressed as 1 - sum(p_i^2).

    Higher values indicate broader distribution across communities.
    """
    return float(1.0 - np.sum(np.square(probabilities)))


def inverse_simpson(probabilities: np.ndarray) -> float:
    """Effective diversity based on the inverse Simpson concentration."""
    denominator = float(np.sum(np.square(probabilities)))

    if denominator <= 0:
        return np.nan

    return float(1.0 / denominator)


def gini_coefficient(values: np.ndarray) -> float:
    """
    Calculate the Gini coefficient for non-negative community counts.

    A higher value indicates greater concentration in a small number of
    communities.
    """
    values = np.asarray(values, dtype=float)

    if np.any(values < 0):
        raise ValueError("Gini coefficient requires non-negative values.")

    if np.allclose(values.sum(), 0):
        return np.nan

    sorted_values = np.sort(values)
    n_values = len(sorted_values)
    cumulative = np.cumsum(sorted_values)

    return float(
        (
            n_values
            + 1
            - 2
            * np.sum(cumulative)
            / cumulative[-1]
        )
        / n_values
    )


def concentration_hhi(probabilities: np.ndarray) -> float:
    """
    Herfindahl-Hirschman concentration index.

    Higher values indicate greater concentration.
    """
    return float(np.sum(np.square(probabilities)))


# =============================================================================
# Matrix construction
# =============================================================================

def build_profile_matrices(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build the three principal company-community matrices.

    Returns
    -------
    counts:
        Raw song counts for each company and community.

    company_shares:
        Community distribution within each company.
        Each row sums to one.

    community_shares:
        Company distribution within each community.
        Each column sums to one.
    """
    communities = sorted(df["community"].unique())

    counts = pd.crosstab(
        df["company"],
        df["community"],
        dropna=False,
    )

    counts = counts.reindex(
        index=COMPANY_ORDER,
        columns=communities,
        fill_value=0,
    )

    counts.index.name = "company"
    counts.columns.name = "community"

    company_shares = counts.div(
        counts.sum(axis=1),
        axis=0,
    )

    community_shares = counts.div(
        counts.sum(axis=0),
        axis=1,
    )

    return counts, company_shares, community_shares


# =============================================================================
# Company profile statistics
# =============================================================================

def calculate_company_diversity(
    df: pd.DataFrame,
    counts: pd.DataFrame,
    company_shares: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate catalogue-level semantic diversity for each company."""
    n_possible_communities = counts.shape[1]
    rows = []

    for company in COMPANY_ORDER:
        subset = df.loc[df["company"] == company].copy()

        count_vector = counts.loc[company].to_numpy(dtype=float)
        probability_vector = (
            company_shares.loc[company]
            .to_numpy(dtype=float)
        )

        occupied_mask = count_vector > 0
        occupied_probabilities = probability_vector[occupied_mask]

        dominant_position = int(np.argmax(probability_vector))
        dominant_community = company_shares.columns[
            dominant_position
        ]

        sorted_positions = np.argsort(probability_vector)[::-1]
        second_position = (
            sorted_positions[1]
            if len(sorted_positions) > 1
            else sorted_positions[0]
        )
        second_community = company_shares.columns[
            second_position
        ]

        top_three_share = float(
            np.sort(probability_vector)[::-1][:3].sum()
        )

        rows.append(
            {
                "company": company,
                "n_songs": int(len(subset)),
                "n_artists": int(
                    subset["artist_normalised"].nunique()
                ),
                "min_year": int(subset["year"].min()),
                "max_year": int(subset["year"].max()),
                "n_communities_occupied": int(
                    occupied_mask.sum()
                ),
                "community_coverage_fraction": float(
                    occupied_mask.sum()
                    / n_possible_communities
                ),
                "shannon_entropy": shannon_entropy(
                    occupied_probabilities
                ),
                "normalised_shannon_entropy": (
                    normalised_shannon_entropy(
                        occupied_probabilities,
                        n_possible_communities,
                    )
                ),
                "effective_n_communities": (
                    effective_number_of_communities(
                        occupied_probabilities
                    )
                ),
                "simpson_diversity": simpson_diversity(
                    occupied_probabilities
                ),
                "inverse_simpson": inverse_simpson(
                    occupied_probabilities
                ),
                "hhi_concentration": concentration_hhi(
                    probability_vector
                ),
                "gini_concentration": gini_coefficient(
                    count_vector
                ),
                "dominant_community": int(
                    dominant_community
                ),
                "dominant_community_share": float(
                    probability_vector[dominant_position]
                ),
                "second_community": int(
                    second_community
                ),
                "second_community_share": float(
                    probability_vector[second_position]
                ),
                "dominance_margin": float(
                    probability_vector[dominant_position]
                    - probability_vector[second_position]
                ),
                "top_three_communities_share": (
                    top_three_share
                ),
                "mean_participation": float(
                    subset[
                        "participation_coefficient"
                    ].mean()
                ),
                "median_participation": float(
                    subset[
                        "participation_coefficient"
                    ].median()
                ),
                "mean_neighbour_entropy": float(
                    subset["community_entropy"].mean()
                ),
                "mean_external_strength_fraction": float(
                    subset[
                        "external_strength_fraction"
                    ].mean()
                ),
                "mean_external_communities_reached": float(
                    subset[
                        "external_communities_reached"
                    ].mean()
                ),
                "mean_centroid_novelty": float(
                    subset["centroid_novelty"].mean()
                ),
                "median_centroid_novelty": float(
                    subset["centroid_novelty"].median()
                ),
                "mean_local_novelty": float(
                    subset["local_novelty"].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Community profile statistics
# =============================================================================

def calculate_community_profiles(
    df: pd.DataFrame,
    counts: pd.DataFrame,
    community_shares: pd.DataFrame,
) -> pd.DataFrame:
    """
    Quantify company concentration and dominance within each community.
    """
    rows = []

    for community in counts.columns:
        subset = df.loc[
            df["community"] == community
        ].copy()

        company_count_vector = (
            counts[community]
            .reindex(COMPANY_ORDER)
            .to_numpy(dtype=float)
        )

        company_probability_vector = (
            community_shares[community]
            .reindex(COMPANY_ORDER)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

        sorted_positions = np.argsort(
            company_probability_vector
        )[::-1]

        dominant_position = sorted_positions[0]
        second_position = sorted_positions[1]

        dominant_company = COMPANY_ORDER[
            dominant_position
        ]
        second_company = COMPANY_ORDER[
            second_position
        ]

        rows.append(
            {
                "community": int(community),
                "n_big4_songs": int(len(subset)),
                "n_big4_artists": int(
                    subset["artist_normalised"].nunique()
                ),
                "n_companies_present": int(
                    np.sum(company_count_vector > 0)
                ),
                "dominant_company": dominant_company,
                "dominant_company_share": float(
                    company_probability_vector[
                        dominant_position
                    ]
                ),
                "second_company": second_company,
                "second_company_share": float(
                    company_probability_vector[
                        second_position
                    ]
                ),
                "dominance_margin": float(
                    company_probability_vector[
                        dominant_position
                    ]
                    - company_probability_vector[
                        second_position
                    ]
                ),
                "company_shannon_entropy": (
                    shannon_entropy(
                        company_probability_vector[
                            company_probability_vector > 0
                        ]
                    )
                ),
                "normalised_company_entropy": (
                    normalised_shannon_entropy(
                        company_probability_vector[
                            company_probability_vector > 0
                        ],
                        len(COMPANY_ORDER),
                    )
                ),
                "company_hhi_concentration": (
                    concentration_hhi(
                        company_probability_vector
                    )
                ),
                "company_gini_concentration": (
                    gini_coefficient(
                        company_count_vector
                    )
                ),
                "mean_year": float(
                    subset["year"].mean()
                ),
                "median_year": float(
                    subset["year"].median()
                ),
                "mean_participation": float(
                    subset[
                        "participation_coefficient"
                    ].mean()
                ),
                "mean_centroid_novelty": float(
                    subset["centroid_novelty"].mean()
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("community")
        .reset_index(drop=True)
    )


# =============================================================================
# Pairwise company fingerprint similarity
# =============================================================================

def cosine_similarity(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """Calculate cosine similarity between two profile vectors."""
    denominator = (
        np.linalg.norm(first)
        * np.linalg.norm(second)
    )

    if denominator <= 0:
        return np.nan

    return float(
        np.dot(first, second)
        / denominator
    )


def calculate_pairwise_company_similarity(
    company_shares: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare the four company semantic fingerprints.

    Metrics
    -------
    cosine_similarity:
        Directional similarity between profile vectors.

    jensen_shannon_distance:
        Symmetric distance between probability distributions.

    l1_distance:
        Total absolute difference between the profile vectors.

    profile_overlap:
        Sum of the community-wise minimum probabilities.
    """
    rows = []

    for first_index, first_company in enumerate(
        COMPANY_ORDER
    ):
        first = (
            company_shares.loc[first_company]
            .to_numpy(dtype=float)
        )

        for second_company in COMPANY_ORDER[
            first_index + 1:
        ]:
            second = (
                company_shares.loc[second_company]
                .to_numpy(dtype=float)
            )

            js_distance = float(
                jensenshannon(
                    first + EPSILON,
                    second + EPSILON,
                    base=2,
                )
            )

            rows.append(
                {
                    "company_1": first_company,
                    "company_2": second_company,
                    "cosine_similarity": (
                        cosine_similarity(
                            first,
                            second,
                        )
                    ),
                    "jensen_shannon_distance": (
                        js_distance
                    ),
                    "jensen_shannon_similarity": (
                        1.0 - js_distance
                    ),
                    "l1_distance": float(
                        np.abs(first - second).sum()
                    ),
                    "profile_overlap": float(
                        np.minimum(first, second).sum()
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_similarity_matrix(
    pairwise: pd.DataFrame,
    value_column: str,
    diagonal_value: float,
) -> pd.DataFrame:
    """Convert long pairwise results into a symmetric matrix."""
    matrix = pd.DataFrame(
        np.nan,
        index=COMPANY_ORDER,
        columns=COMPANY_ORDER,
    )

    np.fill_diagonal(
        matrix.values,
        diagonal_value,
    )

    for row in pairwise.itertuples(index=False):
        matrix.loc[
            row.company_1,
            row.company_2,
        ] = getattr(row, value_column)

        matrix.loc[
            row.company_2,
            row.company_1,
        ] = getattr(row, value_column)

    matrix.index.name = "company"
    return matrix


# =============================================================================
# Long-format outputs
# =============================================================================

def build_long_profile_table(
    counts: pd.DataFrame,
    company_shares: pd.DataFrame,
    community_shares: pd.DataFrame,
) -> pd.DataFrame:
    """Create one long-format table suitable for plotting."""
    rows = []

    for company in COMPANY_ORDER:
        for community in counts.columns:
            rows.append(
                {
                    "company": company,
                    "community": int(community),
                    "n_songs": int(
                        counts.loc[
                            company,
                            community,
                        ]
                    ),
                    "share_within_company": float(
                        company_shares.loc[
                            company,
                            community,
                        ]
                    ),
                    "share_within_community": float(
                        community_shares.loc[
                            company,
                            community,
                        ]
                    ),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# Representative artists and songs
# =============================================================================

def calculate_representatives(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify dominant artists and representative songs for each
    company-community combination.
    """
    artist_counts = (
        df.groupby(
            [
                "company",
                "community",
                "artist_normalised",
            ],
            observed=True,
        )
        .size()
        .rename("n_songs")
        .reset_index()
    )

    artist_counts["artist_rank"] = (
        artist_counts.groupby(
            ["company", "community"],
            observed=True,
        )["n_songs"]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    top_artists = (
        artist_counts.loc[
            artist_counts["artist_rank"] <= 5
        ]
        .sort_values(
            [
                "company",
                "community",
                "artist_rank",
            ]
        )
        .reset_index(drop=True)
    )

    representative_songs = (
        df.sort_values(
            [
                "company",
                "community",
                "centroid_novelty",
                "participation_coefficient",
            ],
            ascending=[
                True,
                True,
                True,
                False,
            ],
        )
        .groupby(
            ["company", "community"],
            observed=True,
        )
        .head(5)
        [
            [
                "company",
                "community",
                "song_uid",
                "title",
                "artist",
                "artist_normalised",
                "year",
                "centroid_novelty",
                "participation_coefficient",
            ]
        ]
        .copy()
    )

    representative_songs["song_rank"] = (
        representative_songs.groupby(
            ["company", "community"],
            observed=True,
        )
        .cumcount()
        + 1
    )

    return top_artists, representative_songs


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Master dataset not found:\n{INPUT_PATH}"
        )

    print("=" * 88)
    print("COMPANY SEMANTIC PROFILE ANALYSIS")
    print("=" * 88)
    print(f"Input: {INPUT_PATH}")

    df = pd.read_parquet(INPUT_PATH)

    validate_input(df)

    df["company"] = pd.Categorical(
        df["company"],
        categories=COMPANY_ORDER,
        ordered=True,
    )

    df["community"] = pd.to_numeric(
        df["community"],
        errors="raise",
    ).astype(int)

    counts, company_shares, community_shares = (
        build_profile_matrices(df)
    )

    company_diversity = calculate_company_diversity(
        df=df,
        counts=counts,
        company_shares=company_shares,
    )

    community_profiles = calculate_community_profiles(
        df=df,
        counts=counts,
        community_shares=community_shares,
    )

    pairwise_similarity = (
        calculate_pairwise_company_similarity(
            company_shares
        )
    )

    cosine_matrix = build_similarity_matrix(
        pairwise=pairwise_similarity,
        value_column="cosine_similarity",
        diagonal_value=1.0,
    )

    js_distance_matrix = build_similarity_matrix(
        pairwise=pairwise_similarity,
        value_column="jensen_shannon_distance",
        diagonal_value=0.0,
    )

    profile_long = build_long_profile_table(
        counts=counts,
        company_shares=company_shares,
        community_shares=community_shares,
    )

    top_artists, representative_songs = (
        calculate_representatives(df)
    )

    # -------------------------------------------------------------------------
    # Save raw matrices
    # -------------------------------------------------------------------------

    counts.to_csv(
        RAW_OUTPUT_DIR
        / "company_by_community_counts.csv"
    )

    company_shares.to_csv(
        RAW_OUTPUT_DIR
        / "company_by_community_shares.csv"
    )

    community_shares.to_csv(
        RAW_OUTPUT_DIR
        / "community_by_company_shares.csv"
    )

    profile_long.to_csv(
        RAW_OUTPUT_DIR
        / "company_community_profiles_long.csv",
        index=False,
    )

    cosine_matrix.to_csv(
        RAW_OUTPUT_DIR
        / "company_profile_cosine_similarity_matrix.csv"
    )

    js_distance_matrix.to_csv(
        RAW_OUTPUT_DIR
        / "company_profile_jensen_shannon_distance_matrix.csv"
    )

    # -------------------------------------------------------------------------
    # Save tables
    # -------------------------------------------------------------------------

    company_diversity.to_csv(
        TABLE_OUTPUT_DIR
        / "company_semantic_diversity.csv",
        index=False,
    )

    community_profiles.to_csv(
        TABLE_OUTPUT_DIR
        / "community_company_concentration.csv",
        index=False,
    )

    pairwise_similarity.to_csv(
        TABLE_OUTPUT_DIR
        / "pairwise_company_profile_similarity.csv",
        index=False,
    )

    top_artists.to_csv(
        TABLE_OUTPUT_DIR
        / "top_artists_by_company_community.csv",
        index=False,
    )

    representative_songs.to_csv(
        TABLE_OUTPUT_DIR
        / "representative_songs_by_company_community.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Publication-oriented compact summary
    # -------------------------------------------------------------------------

    publication_summary_columns = [
        "company",
        "n_songs",
        "n_artists",
        "n_communities_occupied",
        "normalised_shannon_entropy",
        "effective_n_communities",
        "simpson_diversity",
        "hhi_concentration",
        "dominant_community",
        "dominant_community_share",
        "top_three_communities_share",
        "mean_participation",
        "mean_neighbour_entropy",
        "mean_centroid_novelty",
        "mean_local_novelty",
    ]

    company_diversity[
        publication_summary_columns
    ].to_csv(
        TABLE_OUTPUT_DIR
        / "table_company_semantic_profiles.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Save machine-readable report
    # -------------------------------------------------------------------------

    report = {
        "n_songs": int(len(df)),
        "n_artists": int(
            df["artist_normalised"].nunique()
        ),
        "n_companies": int(
            df["company"].nunique()
        ),
        "n_communities_observed": int(
            df["community"].nunique()
        ),
        "community_ids": [
            int(value)
            for value in sorted(
                df["community"].unique()
            )
        ],
        "company_order": COMPANY_ORDER,
        "most_diverse_company_normalised_entropy": (
            str(
                company_diversity.loc[
                    company_diversity[
                        "normalised_shannon_entropy"
                    ].idxmax(),
                    "company",
                ]
            )
        ),
        "most_concentrated_company_hhi": (
            str(
                company_diversity.loc[
                    company_diversity[
                        "hhi_concentration"
                    ].idxmax(),
                    "company",
                ]
            )
        ),
        "most_company_concentrated_community": int(
            community_profiles.loc[
                community_profiles[
                    "dominant_company_share"
                ].idxmax(),
                "community",
            ]
        ),
    }

    with (
        REPORT_OUTPUT_DIR
        / "company_semantic_profiles_report.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------

    print("\n" + "=" * 88)
    print("COMPANY SEMANTIC DIVERSITY")
    print("=" * 88)

    display_columns = [
        "company",
        "n_songs",
        "n_artists",
        "n_communities_occupied",
        "normalised_shannon_entropy",
        "effective_n_communities",
        "hhi_concentration",
        "dominant_community",
        "dominant_community_share",
        "mean_participation",
        "mean_centroid_novelty",
    ]

    print(
        company_diversity[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\n" + "=" * 88)
    print("MOST COMPANY-CONCENTRATED COMMUNITIES")
    print("=" * 88)

    concentrated_display = (
        community_profiles
        .sort_values(
            "dominant_company_share",
            ascending=False,
        )
        .head(10)
        [
            [
                "community",
                "n_big4_songs",
                "dominant_company",
                "dominant_company_share",
                "second_company",
                "second_company_share",
                "dominance_margin",
                "normalised_company_entropy",
            ]
        ]
    )

    print(
        concentrated_display.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\n" + "=" * 88)
    print("PAIRWISE COMPANY PROFILE SIMILARITY")
    print("=" * 88)

    print(
        pairwise_similarity.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\n" + "=" * 88)
    print("SAVED OUTPUTS")
    print("=" * 88)

    print(f"Raw matrices:\n  {RAW_OUTPUT_DIR}")
    print(f"Profile tables:\n  {TABLE_OUTPUT_DIR}")
    print(f"Reports:\n  {REPORT_OUTPUT_DIR}")


if __name__ == "__main__":
    main()