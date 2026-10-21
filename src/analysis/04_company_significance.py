"""
04_company_significance.py

Inferential analysis of corporate semantic identity in K-pop lyrics.

The script performs:

1. Company-community association testing
   - Chi-square test
   - Cramer's V
   - Pearson residuals
   - Adjusted residuals
   - Community-specific company enrichment
   - Artist-block permutation test

2. Artist-level semantic metric comparisons
   - Kruskal-Wallis tests
   - Epsilon-squared effect sizes
   - Pairwise Mann-Whitney U tests
   - Rank-biserial effect sizes
   - Holm correction

3. Artist-block bootstrap uncertainty for catalogue-level diversity
   - Normalised Shannon entropy
   - Effective number of communities
   - HHI concentration
   - Community coverage

All primary resampling is performed at artist level so that prolific artists
do not receive disproportionate inferential weight.

Run from the K-Pop Project root:

    python company_semantics/src/analysis/04_company_significance.py
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import (
    chi2_contingency,
    kruskal,
    mannwhitneyu,
    norm,
)


# =============================================================================
# Paths
# =============================================================================

COMPANY_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = COMPANY_ROOT.parent

INPUT_PATH = (
    COMPANY_ROOT
    / "outputs"
    / "tables"
    / "company_master_analysis_v1.parquet"
)

GLOBAL_COMMUNITY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed_revision1"
    / "lyric_communities_revision1.parquet"
)

OUTPUT_ROOT = (
    COMPANY_ROOT
    / "outputs"
    / "company_significance"
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

METRICS = [
    "participation_coefficient",
    "community_entropy",
    "external_strength_fraction",
    "external_communities_reached",
    "centroid_novelty",
    "local_novelty",
    "centroid_margin",
]

RANDOM_SEED = 20260716

# Use 500 initially for a quick test if needed.
N_ARTIST_PERMUTATIONS = 2000
N_ARTIST_BOOTSTRAPS = 2000

ALPHA = 0.05
EPSILON = 1e-12


# =============================================================================
# Validation
# =============================================================================

REQUIRED_COLUMNS = {
    "song_uid",
    "artist_normalised",
    "company",
    "community",
    *METRICS,
}


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

    artist_company_counts = (
        df.groupby("artist_normalised", observed=True)["company"]
        .nunique()
    )

    inconsistent_artists = artist_company_counts[
        artist_company_counts > 1
    ]

    if not inconsistent_artists.empty:
        raise ValueError(
            "Some artists are assigned to more than one company:\n"
            + "\n".join(inconsistent_artists.index.astype(str))
        )


# =============================================================================
# Multiple-testing correction
# =============================================================================

def holm_adjust(p_values: list[float] | np.ndarray) -> np.ndarray:
    """
    Apply Holm's step-down family-wise error correction.

    Parameters
    ----------
    p_values:
        Raw p-values.

    Returns
    -------
    np.ndarray
        Holm-adjusted p-values in the original order.
    """
    p_values_array = np.asarray(p_values, dtype=float)
    n_tests = len(p_values_array)

    if n_tests == 0:
        return np.array([], dtype=float)

    order = np.argsort(p_values_array)
    sorted_p = p_values_array[order]

    adjusted_sorted = np.empty(n_tests, dtype=float)
    running_maximum = 0.0

    for rank, p_value in enumerate(sorted_p):
        multiplier = n_tests - rank
        adjusted_value = min(1.0, multiplier * p_value)
        running_maximum = max(running_maximum, adjusted_value)
        adjusted_sorted[rank] = running_maximum

    adjusted = np.empty(n_tests, dtype=float)
    adjusted[order] = adjusted_sorted

    return adjusted


# =============================================================================
# Company-community association
# =============================================================================

def cramers_v(
    contingency_table: np.ndarray,
    chi_square: float,
) -> float:
    """
    Calculate Cramer's V for a contingency table.
    """
    n_observations = contingency_table.sum()
    n_rows, n_columns = contingency_table.shape

    denominator = (
        n_observations
        * min(n_rows - 1, n_columns - 1)
    )

    if denominator <= 0:
        return np.nan

    return float(np.sqrt(chi_square / denominator))


def bias_corrected_cramers_v(
    contingency_table: np.ndarray,
    chi_square: float,
) -> float:
    """
    Calculate bias-corrected Cramer's V following Bergsma (2013).
    """
    n = contingency_table.sum()
    rows, columns = contingency_table.shape

    if n <= 1:
        return np.nan

    phi_squared = chi_square / n

    phi_squared_corrected = max(
        0.0,
        phi_squared
        - ((columns - 1) * (rows - 1)) / (n - 1),
    )

    rows_corrected = rows - ((rows - 1) ** 2) / (n - 1)
    columns_corrected = (
        columns
        - ((columns - 1) ** 2) / (n - 1)
    )

    denominator = min(
        columns_corrected - 1,
        rows_corrected - 1,
    )

    if denominator <= 0:
        return np.nan

    return float(
        np.sqrt(phi_squared_corrected / denominator)
    )


def build_contingency_table(
    df: pd.DataFrame,
    communities: list[int],
) -> pd.DataFrame:
    """Build a complete company-community contingency table."""
    table = pd.crosstab(
        df["company"],
        df["community"],
        dropna=False,
    )

    table = table.reindex(
        index=COMPANY_ORDER,
        columns=communities,
        fill_value=0,
    )

    table.index.name = "company"
    table.columns.name = "community"

    return table


def calculate_residuals_and_enrichment(
    observed: pd.DataFrame,
    expected: np.ndarray,
) -> pd.DataFrame:
    """
    Calculate cell-level residuals and enrichment measures.

    Adjusted residuals account for the row and column marginal proportions.
    Under the null hypothesis, they are approximately standard normal.
    """
    observed_values = observed.to_numpy(dtype=float)
    total = observed_values.sum()

    row_totals = observed_values.sum(axis=1)
    column_totals = observed_values.sum(axis=0)

    row_proportions = row_totals / total
    column_proportions = column_totals / total

    overall_company_share = (
        row_totals / total
    )

    rows = []

    for company_index, company in enumerate(observed.index):
        for community_index, community in enumerate(observed.columns):
            observed_count = observed_values[
                company_index,
                community_index,
            ]

            expected_count = expected[
                company_index,
                community_index,
            ]

            if expected_count > 0:
                pearson_residual = (
                    observed_count - expected_count
                ) / np.sqrt(expected_count)
            else:
                pearson_residual = np.nan

            adjusted_denominator = np.sqrt(
                expected_count
                * (1.0 - row_proportions[company_index])
                * (1.0 - column_proportions[community_index])
            )

            if adjusted_denominator > 0:
                adjusted_residual = (
                    observed_count - expected_count
                ) / adjusted_denominator
            else:
                adjusted_residual = np.nan

            p_value = (
                2.0
                * norm.sf(abs(adjusted_residual))
                if np.isfinite(adjusted_residual)
                else np.nan
            )

            community_total = column_totals[
                community_index
            ]

            observed_share_within_community = (
                observed_count / community_total
                if community_total > 0
                else np.nan
            )

            expected_company_share = overall_company_share[
                company_index
            ]

            enrichment_ratio = (
                observed_share_within_community
                / expected_company_share
                if expected_company_share > 0
                else np.nan
            )

            log2_enrichment = (
                np.log2(enrichment_ratio)
                if enrichment_ratio > 0
                else np.nan
            )

            rows.append(
                {
                    "company": company,
                    "community": int(community),
                    "observed_count": int(observed_count),
                    "expected_count": float(expected_count),
                    "pearson_residual": float(pearson_residual),
                    "adjusted_residual": float(adjusted_residual),
                    "raw_p_value": float(p_value),
                    "observed_share_within_community": float(
                        observed_share_within_community
                    ),
                    "overall_company_share": float(
                        expected_company_share
                    ),
                    "enrichment_ratio": float(
                        enrichment_ratio
                    ),
                    "log2_enrichment": float(
                        log2_enrichment
                    )
                    if np.isfinite(log2_enrichment)
                    else np.nan,
                }
            )

    residuals = pd.DataFrame(rows)

    valid_mask = residuals["raw_p_value"].notna()

    residuals.loc[
        valid_mask,
        "holm_adjusted_p",
    ] = holm_adjust(
        residuals.loc[
            valid_mask,
            "raw_p_value",
        ].to_numpy()
    )

    residuals["significant_holm_005"] = (
        residuals["holm_adjusted_p"] < ALPHA
    )

    residuals["direction"] = np.select(
        [
            residuals["adjusted_residual"] > 0,
            residuals["adjusted_residual"] < 0,
        ],
        [
            "overrepresented",
            "underrepresented",
        ],
        default="as_expected",
    )

    return residuals


# =============================================================================
# Artist-block permutation
# =============================================================================

def artist_block_permutation_test(
    df: pd.DataFrame,
    communities: list[int],
    observed_chi_square: float,
    observed_cramers_v: float,
    n_permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    """
    Test company-community association while preserving artist song blocks.

    Company labels are permuted across artists rather than across songs.
    Every song belonging to a given artist retains the same permuted company.
    """
    rng = np.random.default_rng(seed)

    artist_company = (
        df[
            [
                "artist_normalised",
                "company",
            ]
        ]
        .drop_duplicates()
        .sort_values("artist_normalised")
        .reset_index(drop=True)
    )

    artist_names = artist_company[
        "artist_normalised"
    ].to_numpy()

    company_labels = artist_company[
        "company"
    ].astype(str).to_numpy()

    song_artists = df[
        "artist_normalised"
    ].astype(str)

    null_rows = []

    print(
        f"\nRunning {n_permutations:,} artist-block permutations..."
    )

    progress_step = max(1, n_permutations // 10)

    for permutation_index in range(n_permutations):
        permuted_labels = rng.permutation(company_labels)

        artist_to_permuted_company = dict(
            zip(
                artist_names,
                permuted_labels,
            )
        )

        permuted_company = song_artists.map(
            artist_to_permuted_company
        )

        permuted_table = pd.crosstab(
            permuted_company,
            df["community"],
            dropna=False,
        ).reindex(
            index=COMPANY_ORDER,
            columns=communities,
            fill_value=0,
        )

        chi_square, _, _, _ = chi2_contingency(
            permuted_table.to_numpy(),
            correction=False,
        )

        null_cramers_v = cramers_v(
            permuted_table.to_numpy(),
            chi_square,
        )

        null_rows.append(
            {
                "permutation": permutation_index + 1,
                "chi_square": float(chi_square),
                "cramers_v": float(null_cramers_v),
            }
        )

        if (
            (permutation_index + 1) % progress_step == 0
            or permutation_index == n_permutations - 1
        ):
            print(
                f"  Completed {permutation_index + 1:,}/"
                f"{n_permutations:,}"
            )

    null_distribution = pd.DataFrame(null_rows)

    chi_square_p = (
        1
        + np.sum(
            null_distribution["chi_square"]
            >= observed_chi_square
        )
    ) / (n_permutations + 1)

    cramers_v_p = (
        1
        + np.sum(
            null_distribution["cramers_v"]
            >= observed_cramers_v
        )
    ) / (n_permutations + 1)

    summary = {
        "n_permutations": int(n_permutations),
        "n_artists": int(len(artist_company)),
        "observed_chi_square": float(observed_chi_square),
        "permutation_p_chi_square": float(chi_square_p),
        "observed_cramers_v": float(observed_cramers_v),
        "permutation_p_cramers_v": float(cramers_v_p),
        "null_chi_square_mean": float(
            null_distribution["chi_square"].mean()
        ),
        "null_chi_square_std": float(
            null_distribution["chi_square"].std(ddof=1)
        ),
        "null_chi_square_q025": float(
            null_distribution["chi_square"].quantile(0.025)
        ),
        "null_chi_square_q975": float(
            null_distribution["chi_square"].quantile(0.975)
        ),
        "null_cramers_v_mean": float(
            null_distribution["cramers_v"].mean()
        ),
        "null_cramers_v_std": float(
            null_distribution["cramers_v"].std(ddof=1)
        ),
        "null_cramers_v_q025": float(
            null_distribution["cramers_v"].quantile(0.025)
        ),
        "null_cramers_v_q975": float(
            null_distribution["cramers_v"].quantile(0.975)
        ),
    }

    return null_distribution, summary


# =============================================================================
# Artist-level semantic metric comparisons
# =============================================================================

def epsilon_squared(
    h_statistic: float,
    n_observations: int,
    n_groups: int,
) -> float:
    """
    Calculate epsilon-squared effect size for a Kruskal-Wallis test.
    """
    denominator = n_observations - n_groups

    if denominator <= 0:
        return np.nan

    value = (
        h_statistic - n_groups + 1
    ) / denominator

    return float(max(0.0, value))


def rank_biserial_from_u(
    u_statistic: float,
    n_first: int,
    n_second: int,
) -> float:
    """
    Rank-biserial correlation based on the Mann-Whitney U statistic.

    Positive values indicate that the first group tends to have higher values.
    """
    denominator = n_first * n_second

    if denominator <= 0:
        return np.nan

    return float(
        (2.0 * u_statistic / denominator) - 1.0
    )


def build_artist_level_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate song-level semantic metrics to one record per artist."""
    artist_level = (
        df.groupby(
            [
                "company",
                "artist_normalised",
            ],
            observed=True,
            as_index=False,
        )
        .agg(
            n_songs=("song_uid", "size"),
            n_communities=("community", "nunique"),
            mean_participation=(
                "participation_coefficient",
                "mean",
            ),
            mean_community_entropy=(
                "community_entropy",
                "mean",
            ),
            mean_external_strength_fraction=(
                "external_strength_fraction",
                "mean",
            ),
            mean_external_communities_reached=(
                "external_communities_reached",
                "mean",
            ),
            mean_centroid_novelty=(
                "centroid_novelty",
                "mean",
            ),
            mean_local_novelty=(
                "local_novelty",
                "mean",
            ),
            mean_centroid_margin=(
                "centroid_margin",
                "mean",
            ),
        )
    )

    return artist_level


ARTIST_METRIC_MAP = {
    "participation_coefficient": "mean_participation",
    "community_entropy": "mean_community_entropy",
    "external_strength_fraction": (
        "mean_external_strength_fraction"
    ),
    "external_communities_reached": (
        "mean_external_communities_reached"
    ),
    "centroid_novelty": "mean_centroid_novelty",
    "local_novelty": "mean_local_novelty",
    "centroid_margin": "mean_centroid_margin",
}


def artist_level_metric_tests(
    artist_level: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run global and pairwise non-parametric company comparisons.
    """
    global_rows = []
    pairwise_rows = []

    for original_metric, artist_metric in ARTIST_METRIC_MAP.items():
        grouped_values = []

        for company in COMPANY_ORDER:
            values = (
                artist_level.loc[
                    artist_level["company"] == company,
                    artist_metric,
                ]
                .dropna()
                .to_numpy(dtype=float)
            )

            grouped_values.append(values)

        h_statistic, p_value = kruskal(
            *grouped_values,
            nan_policy="omit",
        )

        n_total = sum(
            len(values)
            for values in grouped_values
        )

        global_rows.append(
            {
                "metric": original_metric,
                "artist_level_column": artist_metric,
                "n_artists": int(n_total),
                "n_groups": len(COMPANY_ORDER),
                "kruskal_h": float(h_statistic),
                "degrees_of_freedom": (
                    len(COMPANY_ORDER) - 1
                ),
                "raw_p_value": float(p_value),
                "epsilon_squared": epsilon_squared(
                    h_statistic=h_statistic,
                    n_observations=n_total,
                    n_groups=len(COMPANY_ORDER),
                ),
            }
        )

        metric_pair_rows = []

        for first_company, second_company in combinations(
            COMPANY_ORDER,
            2,
        ):
            first_values = (
                artist_level.loc[
                    artist_level["company"] == first_company,
                    artist_metric,
                ]
                .dropna()
                .to_numpy(dtype=float)
            )

            second_values = (
                artist_level.loc[
                    artist_level["company"] == second_company,
                    artist_metric,
                ]
                .dropna()
                .to_numpy(dtype=float)
            )

            u_statistic, pair_p = mannwhitneyu(
                first_values,
                second_values,
                alternative="two-sided",
                method="auto",
            )

            metric_pair_rows.append(
                {
                    "metric": original_metric,
                    "artist_level_column": artist_metric,
                    "company_1": first_company,
                    "company_2": second_company,
                    "n_1": int(len(first_values)),
                    "n_2": int(len(second_values)),
                    "mean_1": float(np.mean(first_values)),
                    "mean_2": float(np.mean(second_values)),
                    "median_1": float(np.median(first_values)),
                    "median_2": float(np.median(second_values)),
                    "mean_difference_1_minus_2": float(
                        np.mean(first_values)
                        - np.mean(second_values)
                    ),
                    "mann_whitney_u": float(u_statistic),
                    "raw_p_value": float(pair_p),
                    "rank_biserial": (
                        rank_biserial_from_u(
                            u_statistic=u_statistic,
                            n_first=len(first_values),
                            n_second=len(second_values),
                        )
                    ),
                }
            )

        metric_pairwise = pd.DataFrame(metric_pair_rows)

        metric_pairwise["holm_adjusted_p"] = holm_adjust(
            metric_pairwise["raw_p_value"].to_numpy()
        )

        metric_pairwise["significant_holm_005"] = (
            metric_pairwise["holm_adjusted_p"] < ALPHA
        )

        pairwise_rows.append(metric_pairwise)

    global_tests = pd.DataFrame(global_rows)

    global_tests["holm_adjusted_p"] = holm_adjust(
        global_tests["raw_p_value"].to_numpy()
    )

    global_tests["significant_holm_005"] = (
        global_tests["holm_adjusted_p"] < ALPHA
    )

    pairwise_tests = pd.concat(
        pairwise_rows,
        ignore_index=True,
    )

    return global_tests, pairwise_tests


# =============================================================================
# Catalogue diversity bootstrap
# =============================================================================

def shannon_entropy(probabilities: np.ndarray) -> float:
    probabilities = probabilities[
        probabilities > 0
    ]

    if probabilities.size == 0:
        return np.nan

    return float(
        -np.sum(
            probabilities
            * np.log(probabilities)
        )
    )


def calculate_catalogue_diversity(
    community_values: np.ndarray,
    global_communities: list[int],
) -> dict:
    """
    Calculate catalogue-level semantic diversity over the global community set.
    """
    counts = (
        pd.Series(community_values)
        .value_counts()
        .reindex(
            global_communities,
            fill_value=0,
        )
        .to_numpy(dtype=float)
    )

    total = counts.sum()

    if total <= 0:
        return {
            "normalised_shannon_entropy": np.nan,
            "effective_n_communities": np.nan,
            "hhi_concentration": np.nan,
            "community_coverage": np.nan,
            "n_communities_occupied": np.nan,
        }

    probabilities = counts / total
    positive_probabilities = probabilities[
        probabilities > 0
    ]

    entropy = shannon_entropy(
        positive_probabilities
    )

    maximum_entropy = np.log(
        len(global_communities)
    )

    occupied = int(
        np.sum(counts > 0)
    )

    return {
        "normalised_shannon_entropy": float(
            entropy / maximum_entropy
        ),
        "effective_n_communities": float(
            np.exp(entropy)
        ),
        "hhi_concentration": float(
            np.sum(probabilities ** 2)
        ),
        "community_coverage": float(
            occupied / len(global_communities)
        ),
        "n_communities_occupied": occupied,
    }


def bootstrap_company_diversity(
    df: pd.DataFrame,
    global_communities: list[int],
    n_bootstraps: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Bootstrap catalogue diversity by resampling artists within each company.

    Complete artist song blocks are resampled with replacement.
    """
    rng = np.random.default_rng(seed)

    bootstrap_rows = []
    summary_rows = []

    print(
        f"\nRunning {n_bootstraps:,} artist-block bootstraps per company..."
    )

    for company_index, company in enumerate(COMPANY_ORDER):
        subset = df.loc[
            df["company"] == company
        ].copy()

        artist_groups = {
            artist: artist_subset["community"]
            .to_numpy(dtype=int)
            for artist, artist_subset in subset.groupby(
                "artist_normalised",
                observed=True,
            )
        }

        artists = np.array(
            list(artist_groups.keys()),
            dtype=object,
        )

        if len(artists) == 0:
            continue

        original_metrics = calculate_catalogue_diversity(
            subset["community"].to_numpy(dtype=int),
            global_communities,
        )

        progress_step = max(
            1,
            n_bootstraps // 5,
        )

        for bootstrap_index in range(n_bootstraps):
            sampled_artists = rng.choice(
                artists,
                size=len(artists),
                replace=True,
            )

            sampled_communities = np.concatenate(
                [
                    artist_groups[artist]
                    for artist in sampled_artists
                ]
            )

            bootstrap_metrics = (
                calculate_catalogue_diversity(
                    sampled_communities,
                    global_communities,
                )
            )

            bootstrap_rows.append(
                {
                    "company": company,
                    "bootstrap": bootstrap_index + 1,
                    **bootstrap_metrics,
                }
            )

            if (
                (bootstrap_index + 1) % progress_step == 0
                or bootstrap_index == n_bootstraps - 1
            ):
                print(
                    f"  {company}: "
                    f"{bootstrap_index + 1:,}/"
                    f"{n_bootstraps:,}"
                )

        company_bootstrap = pd.DataFrame(
            [
                row
                for row in bootstrap_rows
                if row["company"] == company
            ]
        )

        for metric, point_estimate in original_metrics.items():
            bootstrap_values = (
                company_bootstrap[metric]
                .dropna()
            )

            summary_rows.append(
                {
                    "company": company,
                    "metric": metric,
                    "point_estimate": float(
                        point_estimate
                    ),
                    "bootstrap_mean": float(
                        bootstrap_values.mean()
                    ),
                    "bootstrap_std": float(
                        bootstrap_values.std(ddof=1)
                    ),
                    "ci_low": float(
                        bootstrap_values.quantile(0.025)
                    ),
                    "ci_high": float(
                        bootstrap_values.quantile(0.975)
                    ),
                    "n_bootstraps": int(
                        len(bootstrap_values)
                    ),
                    "n_artists": int(len(artists)),
                }
            )

    return (
        pd.DataFrame(bootstrap_rows),
        pd.DataFrame(summary_rows),
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Master dataset not found:\n{INPUT_PATH}"
        )

    if not GLOBAL_COMMUNITY_PATH.exists():
        raise FileNotFoundError(
            "Global community file not found:\n"
            f"{GLOBAL_COMMUNITY_PATH}"
        )

    print("=" * 92)
    print("COMPANY SIGNIFICANCE AND ROBUSTNESS ANALYSIS")
    print("=" * 92)
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

    global_community_df = pd.read_parquet(
        GLOBAL_COMMUNITY_PATH
    )

    global_communities = sorted(
        global_community_df[
            "community"
        ]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    observed_communities = sorted(
        df["community"]
        .unique()
        .tolist()
    )

    absent_big4_communities = sorted(
        set(global_communities)
        .difference(observed_communities)
    )

    print(f"\nSongs:              {len(df):,}")
    print(
        "Artists:            "
        f"{df['artist_normalised'].nunique():,}"
    )
    print(
        "Global communities: "
        f"{len(global_communities)}"
    )
    print(
        "Big Four occupied:  "
        f"{len(observed_communities)}"
    )
    print(
        "Absent communities: "
        f"{absent_big4_communities}"
    )

    # -------------------------------------------------------------------------
    # Company-community association
    # -------------------------------------------------------------------------

    contingency = build_contingency_table(
        df=df,
        communities=global_communities,
    )

    non_empty_community_mask = (
        contingency.sum(axis=0) > 0
    )

    contingency_for_test = contingency.loc[
        :,
        non_empty_community_mask,
    ]

    chi_square, chi_square_p, degrees_of_freedom, expected = (
        chi2_contingency(
            contingency_for_test.to_numpy(),
            correction=False,
        )
    )

    observed_cramers_v = cramers_v(
        contingency_for_test.to_numpy(),
        chi_square,
    )

    corrected_cramers_v = bias_corrected_cramers_v(
        contingency_for_test.to_numpy(),
        chi_square,
    )

    association_summary = {
        "n_songs": int(len(df)),
        "n_artists": int(
            df["artist_normalised"].nunique()
        ),
        "n_companies": int(
            contingency_for_test.shape[0]
        ),
        "n_nonempty_communities": int(
            contingency_for_test.shape[1]
        ),
        "chi_square": float(chi_square),
        "degrees_of_freedom": int(
            degrees_of_freedom
        ),
        "asymptotic_p_value": float(
            chi_square_p
        ),
        "cramers_v": float(
            observed_cramers_v
        ),
        "bias_corrected_cramers_v": float(
            corrected_cramers_v
        ),
        "minimum_expected_count": float(
            expected.min()
        ),
        "cells_expected_below_5": int(
            np.sum(expected < 5)
        ),
        "absent_big4_communities": (
            absent_big4_communities
        ),
    }

    residuals = calculate_residuals_and_enrichment(
        observed=contingency_for_test,
        expected=expected,
    )

    # -------------------------------------------------------------------------
    # Artist-block permutation
    # -------------------------------------------------------------------------

    permutation_distribution, permutation_summary = (
        artist_block_permutation_test(
            df=df,
            communities=list(
                contingency_for_test.columns
            ),
            observed_chi_square=chi_square,
            observed_cramers_v=observed_cramers_v,
            n_permutations=N_ARTIST_PERMUTATIONS,
            seed=RANDOM_SEED,
        )
    )

    association_summary.update(
        {
            "artist_block_permutation_p_chi_square": (
                permutation_summary[
                    "permutation_p_chi_square"
                ]
            ),
            "artist_block_permutation_p_cramers_v": (
                permutation_summary[
                    "permutation_p_cramers_v"
                ]
            ),
            "null_cramers_v_mean": (
                permutation_summary[
                    "null_cramers_v_mean"
                ]
            ),
            "null_cramers_v_q025": (
                permutation_summary[
                    "null_cramers_v_q025"
                ]
            ),
            "null_cramers_v_q975": (
                permutation_summary[
                    "null_cramers_v_q975"
                ]
            ),
        }
    )

    # -------------------------------------------------------------------------
    # Artist-level metric tests
    # -------------------------------------------------------------------------

    artist_level = build_artist_level_dataset(df)

    global_metric_tests, pairwise_metric_tests = (
        artist_level_metric_tests(
            artist_level
        )
    )

    # -------------------------------------------------------------------------
    # Artist-block catalogue-diversity bootstrap
    # -------------------------------------------------------------------------

    bootstrap_distribution, bootstrap_summary = (
        bootstrap_company_diversity(
            df=df,
            global_communities=global_communities,
            n_bootstraps=N_ARTIST_BOOTSTRAPS,
            seed=RANDOM_SEED + 1,
        )
    )

    # -------------------------------------------------------------------------
    # Save raw outputs
    # -------------------------------------------------------------------------

    contingency.to_csv(
        RAW_OUTPUT_DIR
        / "company_community_contingency_all_global.csv"
    )

    contingency_for_test.to_csv(
        RAW_OUTPUT_DIR
        / "company_community_contingency_nonempty.csv"
    )

    pd.DataFrame(
        expected,
        index=contingency_for_test.index,
        columns=contingency_for_test.columns,
    ).to_csv(
        RAW_OUTPUT_DIR
        / "company_community_expected_counts.csv"
    )

    permutation_distribution.to_csv(
        RAW_OUTPUT_DIR
        / "artist_block_permutation_distribution.csv",
        index=False,
    )

    bootstrap_distribution.to_parquet(
        RAW_OUTPUT_DIR
        / "artist_block_diversity_bootstrap.parquet",
        index=False,
    )

    artist_level.to_csv(
        RAW_OUTPUT_DIR
        / "artist_level_semantic_metrics.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Save publication tables
    # -------------------------------------------------------------------------

    residuals.to_csv(
        TABLE_OUTPUT_DIR
        / "community_company_enrichment.csv",
        index=False,
    )

    global_metric_tests.to_csv(
        TABLE_OUTPUT_DIR
        / "artist_level_global_metric_tests.csv",
        index=False,
    )

    pairwise_metric_tests.to_csv(
        TABLE_OUTPUT_DIR
        / "artist_level_pairwise_metric_tests.csv",
        index=False,
    )

    bootstrap_summary.to_csv(
        TABLE_OUTPUT_DIR
        / "company_diversity_bootstrap_summary.csv",
        index=False,
    )

    significant_enrichment = (
        residuals.loc[
            residuals["significant_holm_005"]
        ]
        .sort_values(
            [
                "company",
                "adjusted_residual",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )

    significant_enrichment.to_csv(
        TABLE_OUTPUT_DIR
        / "significant_company_community_enrichment.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Save reports
    # -------------------------------------------------------------------------

    with (
        REPORT_OUTPUT_DIR
        / "company_community_association.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            association_summary,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    with (
        REPORT_OUTPUT_DIR
        / "artist_block_permutation_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            permutation_summary,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    # -------------------------------------------------------------------------
    # Console report
    # -------------------------------------------------------------------------

    print("\n" + "=" * 92)
    print("COMPANY-COMMUNITY ASSOCIATION")
    print("=" * 92)

    print(
        f"Chi-square:             {chi_square:.4f}"
    )
    print(
        f"Degrees of freedom:     {degrees_of_freedom}"
    )
    print(
        f"Asymptotic p-value:     {chi_square_p:.6g}"
    )
    print(
        f"Cramer's V:             {observed_cramers_v:.4f}"
    )
    print(
        "Bias-corrected V:      "
        f"{corrected_cramers_v:.4f}"
    )
    print(
        "Minimum expected count:"
        f" {expected.min():.4f}"
    )
    print(
        "Expected cells < 5:    "
        f"{np.sum(expected < 5)}"
    )

    print("\nArtist-block permutation:")
    print(
        "  p (chi-square):       "
        f"{permutation_summary['permutation_p_chi_square']:.6g}"
    )
    print(
        "  p (Cramer's V):       "
        f"{permutation_summary['permutation_p_cramers_v']:.6g}"
    )
    print(
        "  Null V mean:          "
        f"{permutation_summary['null_cramers_v_mean']:.4f}"
    )
    print(
        "  Null V 95% interval:  "
        f"[{permutation_summary['null_cramers_v_q025']:.4f}, "
        f"{permutation_summary['null_cramers_v_q975']:.4f}]"
    )

    print("\n" + "=" * 92)
    print("SIGNIFICANT COMPANY-COMMUNITY ENRICHMENTS")
    print("=" * 92)

    if significant_enrichment.empty:
        print("No cells remained significant after Holm correction.")
    else:
        print(
            significant_enrichment[
                [
                    "company",
                    "community",
                    "observed_count",
                    "expected_count",
                    "adjusted_residual",
                    "enrichment_ratio",
                    "holm_adjusted_p",
                    "direction",
                ]
            ]
            .to_string(
                index=False,
                float_format=lambda value: f"{value:.4f}",
            )
        )

    print("\n" + "=" * 92)
    print("ARTIST-LEVEL GLOBAL METRIC TESTS")
    print("=" * 92)

    print(
        global_metric_tests[
            [
                "metric",
                "n_artists",
                "kruskal_h",
                "raw_p_value",
                "holm_adjusted_p",
                "epsilon_squared",
                "significant_holm_005",
            ]
        ]
        .sort_values("holm_adjusted_p")
        .to_string(
            index=False,
            float_format=lambda value: f"{value:.6g}",
        )
    )

    print("\n" + "=" * 92)
    print("SIGNIFICANT ARTIST-LEVEL PAIRWISE TESTS")
    print("=" * 92)

    significant_pairwise = (
        pairwise_metric_tests.loc[
            pairwise_metric_tests[
                "significant_holm_005"
            ]
        ]
        .sort_values(
            [
                "metric",
                "holm_adjusted_p",
            ]
        )
    )

    if significant_pairwise.empty:
        print("No pairwise tests remained significant after correction.")
    else:
        print(
            significant_pairwise[
                [
                    "metric",
                    "company_1",
                    "company_2",
                    "mean_1",
                    "mean_2",
                    "mean_difference_1_minus_2",
                    "holm_adjusted_p",
                    "rank_biserial",
                ]
            ]
            .to_string(
                index=False,
                float_format=lambda value: f"{value:.6g}",
            )
        )

    print("\n" + "=" * 92)
    print("CATALOGUE DIVERSITY BOOTSTRAP")
    print("=" * 92)

    entropy_bootstrap = (
        bootstrap_summary.loc[
            bootstrap_summary["metric"]
            == "normalised_shannon_entropy"
        ]
    )

    print(
        entropy_bootstrap[
            [
                "company",
                "point_estimate",
                "bootstrap_mean",
                "ci_low",
                "ci_high",
                "n_artists",
            ]
        ]
        .to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\n" + "=" * 92)
    print("SAVED OUTPUTS")
    print("=" * 92)

    print(f"Raw outputs:\n  {RAW_OUTPUT_DIR}")
    print(f"Tables:\n  {TABLE_OUTPUT_DIR}")
    print(f"Reports:\n  {REPORT_OUTPUT_DIR}")


if __name__ == "__main__":
    main()