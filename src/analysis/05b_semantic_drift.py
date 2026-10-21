#!/usr/bin/env python3
"""
=========================================================
WP5 - Temporal Dynamics
05b_semantic_drift.py

Computes semantic drift between successive available
rolling 5-year company fingerprints.

The script preserves all observed transitions, but clearly
distinguishes genuinely consecutive rolling-window
transitions from transitions spanning missing windows.

For each transition, it calculates:

    1. Jensen-Shannon distance
    2. Jensen-Shannon divergence
    3. Cosine similarity
    4. Cosine distance
    5. Euclidean distance
    6. Annualised Jensen-Shannon distance
    7. Annualised Jensen-Shannon divergence
    8. Whether the transition is consecutive

Inferential company comparisons are performed only on
consecutive one-year rolling transitions.

Input:
    outputs/temporal/rolling5/community_profiles.parquet

Outputs:
    outputs/temporal/rolling5/
        semantic_drift.parquet
        semantic_drift_consecutive.parquet
        semantic_drift_irregular.parquet
        semantic_drift_summary_all.csv
        semantic_drift_summary_consecutive.csv
        semantic_drift_rankings_consecutive.csv
        semantic_drift_kruskal_consecutive.csv
        semantic_drift_pairwise_consecutive.csv
=========================================================
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.spatial.distance import (
    cosine,
    euclidean,
    jensenshannon,
)
from scipy.stats import (
    kruskal,
    mannwhitneyu,
)
from statsmodels.stats.multitest import multipletests


# --------------------------------------------------------
# Configuration
# --------------------------------------------------------

SIGNIFICANCE_LEVEL = 0.05

EPSILON = 1e-12

EXPECTED_START_YEAR_GAP = 1


# --------------------------------------------------------
# Paths
# --------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "outputs"
    / "temporal"
    / "rolling5"
    / "community_profiles.parquet"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "temporal"
    / "rolling5"
)

ALL_DRIFT_FILE = (
    OUTPUT_DIR
    / "semantic_drift.parquet"
)

CONSECUTIVE_DRIFT_FILE = (
    OUTPUT_DIR
    / "semantic_drift_consecutive.parquet"
)

IRREGULAR_DRIFT_FILE = (
    OUTPUT_DIR
    / "semantic_drift_irregular.parquet"
)

SUMMARY_ALL_FILE = (
    OUTPUT_DIR
    / "semantic_drift_summary_all.csv"
)

SUMMARY_CONSECUTIVE_FILE = (
    OUTPUT_DIR
    / "semantic_drift_summary_consecutive.csv"
)

RANKING_FILE = (
    OUTPUT_DIR
    / "semantic_drift_rankings_consecutive.csv"
)

KRUSKAL_FILE = (
    OUTPUT_DIR
    / "semantic_drift_kruskal_consecutive.csv"
)

PAIRWISE_FILE = (
    OUTPUT_DIR
    / "semantic_drift_pairwise_consecutive.csv"
)


# --------------------------------------------------------
# Validation
# --------------------------------------------------------

def validate_input(
    dataframe: pd.DataFrame,
    community_columns: list[str],
) -> None:
    """
    Validate the temporal company-profile dataframe.

    Parameters
    ----------
    dataframe:
        Input temporal profile dataframe.

    community_columns:
        Columns containing community proportions.

    Raises
    ------
    ValueError
        If required columns are missing, the dataframe is
        empty, or no community columns are present.
    """

    required_columns = {
        "company",
        "window",
        "start_year",
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
            "The input temporal profile dataframe is empty."
        )

    if not community_columns:
        raise ValueError(
            "No columns beginning with 'community_' were "
            "found in the input file."
        )

    if dataframe["company"].isna().any():
        raise ValueError(
            "The input file contains missing company names."
        )

    if dataframe["window"].isna().any():
        raise ValueError(
            "The input file contains missing window labels."
        )

    if dataframe["start_year"].isna().any():
        raise ValueError(
            "The input file contains missing start years."
        )


def validate_company_windows(
    company_df: pd.DataFrame,
    company: str,
) -> None:
    """
    Check that a company does not contain duplicate start
    years or duplicate window labels.
    """

    duplicate_start_years = (
        company_df["start_year"]
        .duplicated(
            keep=False
        )
    )

    if duplicate_start_years.any():
        duplicated_values = (
            company_df.loc[
                duplicate_start_years,
                "start_year",
            ]
            .astype(int)
            .tolist()
        )

        raise ValueError(
            f"Duplicate start years detected for company "
            f"'{company}': {duplicated_values}"
        )

    duplicate_windows = (
        company_df["window"]
        .duplicated(
            keep=False
        )
    )

    if duplicate_windows.any():
        duplicated_values = (
            company_df.loc[
                duplicate_windows,
                "window",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            f"Duplicate window labels detected for company "
            f"'{company}': {duplicated_values}"
        )


# --------------------------------------------------------
# Probability utilities
# --------------------------------------------------------

def normalise_probability_vector(
    values: np.ndarray,
    company: str,
    window: str,
) -> np.ndarray:
    """
    Convert a fingerprint vector into a valid probability
    distribution.

    Small negative floating-point values are clipped to
    zero. Missing and infinite values are rejected.
    """

    vector = np.asarray(
        values,
        dtype=float,
    )

    if not np.all(
        np.isfinite(vector)
    ):
        raise ValueError(
            "Non-finite community values detected for "
            f"company '{company}', window '{window}'."
        )

    vector = np.clip(
        vector,
        a_min=0.0,
        a_max=None,
    )

    total = float(
        vector.sum()
    )

    if total <= EPSILON:
        raise ValueError(
            "Community profile has a zero or near-zero sum "
            f"for company '{company}', window '{window}'."
        )

    return vector / total


def safe_cosine_similarity(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """
    Calculate cosine similarity and protect against
    non-finite numerical results.
    """

    similarity = (
        1.0
        - cosine(
            first,
            second,
        )
    )

    if not np.isfinite(
        similarity
    ):
        return np.nan

    return float(
        np.clip(
            similarity,
            -1.0,
            1.0,
        )
    )


# --------------------------------------------------------
# Statistical utilities
# --------------------------------------------------------

def safe_coefficient_of_variation(
    standard_deviation: pd.Series,
    mean: pd.Series,
) -> pd.Series:
    """
    Calculate the coefficient of variation without dividing
    by zero.
    """

    valid_mean = mean.where(
        mean.abs() > EPSILON
    )

    return standard_deviation.divide(
        valid_mean
    )


def rank_biserial_from_u(
    u_statistic: float,
    n_first: int,
    n_second: int,
) -> float:
    """
    Calculate rank-biserial correlation from a Mann-Whitney
    U statistic.

    Positive values indicate that company_1 tends to have
    larger drift values than company_2.

    Formula
    -------
    r_rb = (2U / (n1 * n2)) - 1
    """

    denominator = (
        n_first
        * n_second
    )

    if denominator == 0:
        return np.nan

    effect_size = (
        (
            2.0
            * float(u_statistic)
        )
        / float(denominator)
    ) - 1.0

    return float(
        np.clip(
            effect_size,
            -1.0,
            1.0,
        )
    )


def classify_effect_size(
    effect_size: float,
) -> str:
    """
    Classify the absolute rank-biserial effect magnitude.
    """

    if pd.isna(
        effect_size
    ):
        return "undefined"

    absolute_effect = abs(
        effect_size
    )

    if absolute_effect < 0.10:
        return "negligible"

    if absolute_effect < 0.30:
        return "small"

    if absolute_effect < 0.50:
        return "medium"

    return "large"


# --------------------------------------------------------
# Summary construction
# --------------------------------------------------------

def build_summary(
    drift_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build company-level semantic-drift summary statistics.
    """

    summary_columns = [
        "company",
        "n_transitions",
        "mean_year_gap",
        "max_year_gap",
        "mean_js",
        "median_js",
        "std_js",
        "cv_js",
        "min_js",
        "q1_js",
        "q3_js",
        "iqr_js",
        "max_js",
        "mean_js_distance",
        "mean_annualised_js",
        "median_annualised_js",
        "mean_cosine",
        "mean_cosine_distance",
        "mean_euclidean",
    ]

    if drift_df.empty:
        return pd.DataFrame(
            columns=summary_columns
        )

    summary = (
        drift_df
        .groupby(
            "company",
            as_index=False,
        )
        .agg(
            n_transitions=(
                "js_divergence",
                "count",
            ),
            mean_year_gap=(
                "year_gap",
                "mean",
            ),
            max_year_gap=(
                "year_gap",
                "max",
            ),
            mean_js=(
                "js_divergence",
                "mean",
            ),
            median_js=(
                "js_divergence",
                "median",
            ),
            std_js=(
                "js_divergence",
                "std",
            ),
            min_js=(
                "js_divergence",
                "min",
            ),
            max_js=(
                "js_divergence",
                "max",
            ),
            q1_js=(
                "js_divergence",
                lambda values: values.quantile(
                    0.25
                ),
            ),
            q3_js=(
                "js_divergence",
                lambda values: values.quantile(
                    0.75
                ),
            ),
            mean_js_distance=(
                "js_distance",
                "mean",
            ),
            mean_annualised_js=(
                "annualised_js_divergence",
                "mean",
            ),
            median_annualised_js=(
                "annualised_js_divergence",
                "median",
            ),
            mean_cosine=(
                "cosine_similarity",
                "mean",
            ),
            mean_cosine_distance=(
                "cosine_distance",
                "mean",
            ),
            mean_euclidean=(
                "euclidean_distance",
                "mean",
            ),
        )
    )

    summary["iqr_js"] = (
        summary["q3_js"]
        - summary["q1_js"]
    )

    summary["cv_js"] = (
        safe_coefficient_of_variation(
            standard_deviation=summary[
                "std_js"
            ],
            mean=summary[
                "mean_js"
            ],
        )
    )

    return summary[
        summary_columns
    ]


# --------------------------------------------------------
# Global statistical test
# --------------------------------------------------------

def run_kruskal_test(
    drift_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run a Kruskal-Wallis test across company drift
    distributions.

    This function is intended to receive consecutive
    transitions only.
    """

    groups = []

    company_names = []

    for company in sorted(
        drift_df["company"].unique()
    ):

        values = (
            drift_df.loc[
                drift_df["company"] == company,
                "js_divergence",
            ]
            .dropna()
            .to_numpy(dtype=float)
        )

        if len(values) > 0:
            groups.append(
                values
            )

            company_names.append(
                company
            )

    result_columns = [
        "test",
        "metric",
        "transition_subset",
        "n_companies",
        "companies",
        "n_observations",
        "H_statistic",
        "p_value",
        "alpha",
        "significant",
    ]

    if len(groups) < 2:
        return pd.DataFrame(
            [
                {
                    "test": "Kruskal-Wallis",
                    "metric": "js_divergence",
                    "transition_subset": (
                        "consecutive_only"
                    ),
                    "n_companies": len(
                        groups
                    ),
                    "companies": "|".join(
                        company_names
                    ),
                    "n_observations": sum(
                        len(group)
                        for group in groups
                    ),
                    "H_statistic": np.nan,
                    "p_value": np.nan,
                    "alpha": (
                        SIGNIFICANCE_LEVEL
                    ),
                    "significant": False,
                }
            ],
            columns=result_columns,
        )

    h_statistic, p_value = kruskal(
        *groups,
        nan_policy="omit",
    )

    return pd.DataFrame(
        [
            {
                "test": "Kruskal-Wallis",
                "metric": "js_divergence",
                "transition_subset": (
                    "consecutive_only"
                ),
                "n_companies": len(
                    groups
                ),
                "companies": "|".join(
                    company_names
                ),
                "n_observations": sum(
                    len(group)
                    for group in groups
                ),
                "H_statistic": float(
                    h_statistic
                ),
                "p_value": float(
                    p_value
                ),
                "alpha": (
                    SIGNIFICANCE_LEVEL
                ),
                "significant": bool(
                    p_value
                    < SIGNIFICANCE_LEVEL
                ),
            }
        ],
        columns=result_columns,
    )


# --------------------------------------------------------
# Pairwise statistical tests
# --------------------------------------------------------

def run_pairwise_tests(
    drift_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run pairwise Mann-Whitney U tests using consecutive
    transitions only.
    """

    output_columns = [
        "company_1",
        "company_2",
        "transition_subset",
        "n_1",
        "n_2",
        "mean_1",
        "mean_2",
        "median_1",
        "median_2",
        "median_difference_1_minus_2",
        "U_statistic",
        "p_value",
        "holm_p",
        "significant",
        "rank_biserial",
        "absolute_rank_biserial",
        "effect_magnitude",
        "effect_direction",
    ]

    companies = sorted(
        drift_df["company"].unique()
    )

    pairwise_rows = []

    for company_1, company_2 in combinations(
        companies,
        2,
    ):

        first_values = (
            drift_df.loc[
                drift_df["company"] == company_1,
                "js_divergence",
            ]
            .dropna()
            .to_numpy(dtype=float)
        )

        second_values = (
            drift_df.loc[
                drift_df["company"] == company_2,
                "js_divergence",
            ]
            .dropna()
            .to_numpy(dtype=float)
        )

        if (
            len(first_values) == 0
            or len(second_values) == 0
        ):
            continue

        u_statistic, p_value = mannwhitneyu(
            first_values,
            second_values,
            alternative="two-sided",
            method="auto",
        )

        rank_biserial = rank_biserial_from_u(
            u_statistic=u_statistic,
            n_first=len(first_values),
            n_second=len(second_values),
        )

        mean_1 = float(
            np.mean(first_values)
        )

        mean_2 = float(
            np.mean(second_values)
        )

        median_1 = float(
            np.median(first_values)
        )

        median_2 = float(
            np.median(second_values)
        )

        if rank_biserial > 0:
            effect_direction = company_1
        elif rank_biserial < 0:
            effect_direction = company_2
        else:
            effect_direction = "no_direction"

        pairwise_rows.append(
            {
                "company_1": company_1,
                "company_2": company_2,
                "transition_subset": (
                    "consecutive_only"
                ),
                "n_1": len(
                    first_values
                ),
                "n_2": len(
                    second_values
                ),
                "mean_1": mean_1,
                "mean_2": mean_2,
                "median_1": median_1,
                "median_2": median_2,
                "median_difference_1_minus_2": (
                    median_1
                    - median_2
                ),
                "U_statistic": float(
                    u_statistic
                ),
                "p_value": float(
                    p_value
                ),
                "rank_biserial": (
                    rank_biserial
                ),
                "absolute_rank_biserial": (
                    abs(rank_biserial)
                ),
                "effect_magnitude": (
                    classify_effect_size(
                        rank_biserial
                    )
                ),
                "effect_direction": (
                    effect_direction
                ),
            }
        )

    if not pairwise_rows:
        return pd.DataFrame(
            columns=output_columns
        )

    pairwise = pd.DataFrame(
        pairwise_rows
    )

    rejected, holm_p_values, _, _ = (
        multipletests(
            pairwise["p_value"],
            alpha=SIGNIFICANCE_LEVEL,
            method="holm",
        )
    )

    pairwise["holm_p"] = (
        holm_p_values
    )

    pairwise["significant"] = (
        rejected.astype(bool)
    )

    pairwise = pairwise[
        output_columns
    ]

    return (
        pairwise
        .sort_values(
            [
                "holm_p",
                "absolute_rank_biserial",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )


# --------------------------------------------------------
# Main analysis
# --------------------------------------------------------

def main() -> None:
    """
    Run the complete semantic-drift analysis.
    """

    print("=" * 70)
    print("WP5 : SEMANTIC DRIFT")
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
    # Load temporal profiles
    # ----------------------------------------------------

    profiles = pd.read_parquet(
        INPUT_FILE
    )

    community_cols = sorted(
        column
        for column in profiles.columns
        if column.startswith(
            "community_"
        )
    )

    validate_input(
        dataframe=profiles,
        community_columns=community_cols,
    )

    profiles = profiles.copy()

    profiles["company"] = (
        profiles["company"]
        .astype(str)
    )

    profiles["window"] = (
        profiles["window"]
        .astype(str)
    )

    profiles["start_year"] = pd.to_numeric(
        profiles["start_year"],
        errors="raise",
    ).astype(int)

    companies = sorted(
        profiles["company"].unique()
    )

    print()
    print(f"Input file          : {INPUT_FILE}")
    print(f"Rows loaded         : {len(profiles)}")
    print(f"Companies found     : {len(companies)}")
    print(f"Community dimensions: {len(community_cols)}")

    drift_rows = []

    # ----------------------------------------------------
    # Process each company
    # ----------------------------------------------------

    for company in companies:

        company_df = (
            profiles.loc[
                profiles["company"] == company
            ]
            .sort_values(
                [
                    "start_year",
                    "window",
                ]
            )
            .reset_index(drop=True)
        )

        validate_company_windows(
            company_df=company_df,
            company=company,
        )

        if len(company_df) < 2:
            print(
                "Skipping company with fewer than two "
                f"temporal windows: {company}"
            )
            continue

        for index in range(
            len(company_df) - 1
        ):

            previous = company_df.iloc[
                index
            ]

            current = company_df.iloc[
                index + 1
            ]

            previous_window = str(
                previous["window"]
            )

            current_window = str(
                current["window"]
            )

            from_start = int(
                previous["start_year"]
            )

            to_start = int(
                current["start_year"]
            )

            year_gap = (
                to_start
                - from_start
            )

            if year_gap <= 0:
                raise ValueError(
                    "Non-positive temporal gap detected "
                    f"for company '{company}': "
                    f"{previous_window} -> {current_window}"
                )

            is_consecutive = bool(
                year_gap
                == EXPECTED_START_YEAR_GAP
            )

            p = normalise_probability_vector(
                values=previous[
                    community_cols
                ].to_numpy(dtype=float),
                company=company,
                window=previous_window,
            )

            q = normalise_probability_vector(
                values=current[
                    community_cols
                ].to_numpy(dtype=float),
                company=company,
                window=current_window,
            )

            # SciPy returns Jensen-Shannon distance,
            # which is the square root of divergence.
            js_distance = float(
                jensenshannon(
                    p,
                    q,
                    base=2,
                )
            )

            js_divergence = float(
                js_distance ** 2
            )

            cosine_similarity = (
                safe_cosine_similarity(
                    p,
                    q,
                )
            )

            cosine_distance = (
                1.0
                - cosine_similarity
                if np.isfinite(
                    cosine_similarity
                )
                else np.nan
            )

            euclidean_distance = float(
                euclidean(
                    p,
                    q,
                )
            )

            annualised_js_distance = (
                js_distance
                / year_gap
            )

            annualised_js_divergence = (
                js_divergence
                / year_gap
            )

            annualised_euclidean_distance = (
                euclidean_distance
                / year_gap
            )

            drift_rows.append(
                {
                    "company": company,
                    "from_window": (
                        previous_window
                    ),
                    "to_window": (
                        current_window
                    ),
                    "from_start": (
                        from_start
                    ),
                    "to_start": (
                        to_start
                    ),
                    "year_gap": (
                        year_gap
                    ),
                    "is_consecutive": (
                        is_consecutive
                    ),
                    "transition_type": (
                        "consecutive"
                        if is_consecutive
                        else "irregular"
                    ),
                    "js_distance": (
                        js_distance
                    ),
                    "js_divergence": (
                        js_divergence
                    ),
                    "annualised_js_distance": (
                        annualised_js_distance
                    ),
                    "annualised_js_divergence": (
                        annualised_js_divergence
                    ),
                    "cosine_similarity": (
                        cosine_similarity
                    ),
                    "cosine_distance": (
                        cosine_distance
                    ),
                    "euclidean_distance": (
                        euclidean_distance
                    ),
                    "annualised_euclidean_distance": (
                        annualised_euclidean_distance
                    ),
                }
            )

    drift = pd.DataFrame(
        drift_rows
    )

    if drift.empty:
        raise ValueError(
            "No semantic-drift transitions could be "
            "constructed. Each company must have at least "
            "two temporal windows."
        )

    drift = (
        drift
        .sort_values(
            [
                "company",
                "from_start",
                "to_start",
            ]
        )
        .reset_index(drop=True)
    )

    consecutive_drift = (
        drift.loc[
            drift["is_consecutive"]
        ]
        .copy()
        .reset_index(drop=True)
    )

    irregular_drift = (
        drift.loc[
            ~drift["is_consecutive"]
        ]
        .copy()
        .reset_index(drop=True)
    )

    if consecutive_drift.empty:
        raise ValueError(
            "No consecutive one-year rolling-window "
            "transitions were found."
        )

    # ----------------------------------------------------
    # Summary statistics
    # ----------------------------------------------------

    summary_all = build_summary(
        drift_df=drift
    )

    summary_consecutive = build_summary(
        drift_df=consecutive_drift
    )

    ranking = (
        summary_consecutive
        .sort_values(
            [
                "mean_js",
                "median_js",
                "max_js",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    ranking.insert(
        0,
        "rank",
        ranking.index + 1,
    )

    # ----------------------------------------------------
    # Inferential statistics
    # Consecutive transitions only
    # ----------------------------------------------------

    kruskal_result = run_kruskal_test(
        drift_df=consecutive_drift
    )

    pairwise = run_pairwise_tests(
        drift_df=consecutive_drift
    )

    # ----------------------------------------------------
    # Save outputs
    # ----------------------------------------------------

    drift.to_parquet(
        ALL_DRIFT_FILE,
        index=False,
    )

    consecutive_drift.to_parquet(
        CONSECUTIVE_DRIFT_FILE,
        index=False,
    )

    irregular_drift.to_parquet(
        IRREGULAR_DRIFT_FILE,
        index=False,
    )

    summary_all.to_csv(
        SUMMARY_ALL_FILE,
        index=False,
    )

    summary_consecutive.to_csv(
        SUMMARY_CONSECUTIVE_FILE,
        index=False,
    )

    ranking.to_csv(
        RANKING_FILE,
        index=False,
    )

    kruskal_result.to_csv(
        KRUSKAL_FILE,
        index=False,
    )

    pairwise.to_csv(
        PAIRWISE_FILE,
        index=False,
    )

    # ----------------------------------------------------
    # Console report
    # ----------------------------------------------------

    analysed_companies = sorted(
        drift["company"].unique()
    )

    consecutive_companies = sorted(
        consecutive_drift[
            "company"
        ].unique()
    )

    significant_pairs = (
        int(
            pairwise[
                "significant"
            ].sum()
        )
        if not pairwise.empty
        else 0
    )

    print()
    print("=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)

    print(
        f"Companies analysed       : "
        f"{len(analysed_companies)}"
    )

    print(
        f"All transitions          : "
        f"{len(drift)}"
    )

    print(
        f"Consecutive transitions  : "
        f"{len(consecutive_drift)}"
    )

    print(
        f"Irregular transitions    : "
        f"{len(irregular_drift)}"
    )

    print(
        f"Inferential companies    : "
        f"{len(consecutive_companies)}"
    )

    print(
        f"Pairwise tests            : "
        f"{len(pairwise)}"
    )

    print(
        f"Significant pairs         : "
        f"{significant_pairs}"
    )

    # ----------------------------------------------------
    # Irregular transitions
    # ----------------------------------------------------

    print()
    print("=" * 70)
    print("IRREGULAR TRANSITIONS")
    print("=" * 70)

    if irregular_drift.empty:
        print(
            "No irregular transitions were detected."
        )

    else:
        irregular_display_columns = [
            "company",
            "from_window",
            "to_window",
            "from_start",
            "to_start",
            "year_gap",
            "js_divergence",
            "annualised_js_divergence",
        ]

        print(
            irregular_drift[
                irregular_display_columns
            ].to_string(
                index=False
            )
        )

    # ----------------------------------------------------
    # Kruskal-Wallis
    # ----------------------------------------------------

    print()
    print("=" * 70)
    print("KRUSKAL-WALLIS TEST")
    print("CONSECUTIVE TRANSITIONS ONLY")
    print("=" * 70)

    kruskal_row = (
        kruskal_result.iloc[0]
    )

    if pd.notna(
        kruskal_row["H_statistic"]
    ):
        print(
            "H statistic : "
            f"{kruskal_row['H_statistic']:.4f}"
        )

        print(
            "p-value     : "
            f"{kruskal_row['p_value']:.8g}"
        )

        print(
            "Significant : "
            f"{kruskal_row['significant']}"
        )

    else:
        print(
            "The Kruskal-Wallis test could not be "
            "performed because fewer than two valid "
            "company groups were available."
        )

    # ----------------------------------------------------
    # Drift preview
    # ----------------------------------------------------

    print()
    print("=" * 70)
    print("ALL-TRANSITION DRIFT PREVIEW")
    print("=" * 70)

    preview_columns = [
        "company",
        "from_window",
        "to_window",
        "year_gap",
        "is_consecutive",
        "js_distance",
        "js_divergence",
        "annualised_js_divergence",
        "cosine_similarity",
        "euclidean_distance",
    ]

    print(
        drift[
            preview_columns
        ]
        .head()
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------
    # Consecutive summary
    # ----------------------------------------------------

    print()
    print("=" * 70)
    print("CONSECUTIVE-TRANSITION SUMMARY")
    print("=" * 70)

    print(
        summary_consecutive.to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("CONSECUTIVE-TRANSITION RANKING")
    print("=" * 70)

    print(
        ranking.to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("PAIRWISE COMPARISONS")
    print("CONSECUTIVE TRANSITIONS ONLY")
    print("=" * 70)

    if pairwise.empty:
        print(
            "No pairwise comparisons were available."
        )
    else:
        print(
            pairwise.to_string(
                index=False
            )
        )

    print()
    print("=" * 70)
    print("FILES WRITTEN")
    print("=" * 70)

    print(ALL_DRIFT_FILE)
    print(CONSECUTIVE_DRIFT_FILE)
    print(IRREGULAR_DRIFT_FILE)
    print(SUMMARY_ALL_FILE)
    print(SUMMARY_CONSECUTIVE_FILE)
    print(RANKING_FILE)
    print(KRUSKAL_FILE)
    print(PAIRWISE_FILE)

    print("=" * 70)


if __name__ == "__main__":
    main()