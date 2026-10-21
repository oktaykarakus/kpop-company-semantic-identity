"""
WP6e: Temporal-window sensitivity analysis.

This script tests whether company-level temporal conclusions depend on the
choice of temporal window. It compares:

1. Annual windows.
2. Rolling 3-year windows.
3. Rolling 5-year windows.
4. Rolling 7-year windows.

For every valid window, the script reconstructs company community profiles,
pairwise company distances, company distinctiveness rankings, within-company
profile drift, and cross-company convergence. It then compares conclusions
across window definitions.

The authoritative input dataset is:

    outputs/tables/company_master_analysis_v1.parquet

Only rows satisfying analysis_eligible == True are analysed.
"""

from __future__ import annotations

SCRIPT_VERSION = "2026-07-18-final-3"

from itertools import combinations
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
    / "temporal_window_sensitivity"
)

WINDOW_PROFILES_PATH = OUTPUT_DIR / "temporal_window_profiles.csv"
WINDOW_PAIRS_PATH = OUTPUT_DIR / "temporal_window_pairs.csv"
WINDOW_RANKINGS_PATH = OUTPUT_DIR / "temporal_window_rankings.csv"
WINDOW_DRIFT_PATH = OUTPUT_DIR / "temporal_window_drift.csv"
WINDOW_CONVERGENCE_PATH = OUTPUT_DIR / "temporal_window_convergence.csv"
WINDOW_SUMMARY_PATH = OUTPUT_DIR / "window_sensitivity.csv"
COMPANY_SUMMARY_PATH = OUTPUT_DIR / "temporal_company_sensitivity.csv"
PAIR_SUMMARY_PATH = OUTPUT_DIR / "temporal_pair_sensitivity.csv"
PUBLICATION_SUMMARY_PATH = OUTPUT_DIR / "temporal_window_publication_summary.csv"

WINDOW_SPECS = {
    "annual": 1,
    "rolling_3_year": 3,
    "rolling_5_year": 5,
    "rolling_7_year": 7,
}

REFERENCE_WINDOW = "rolling_5_year"
MIN_SONGS_PER_COMPANY_WINDOW = 5
MIN_COMPANIES_PER_WINDOW = 2
EPSILON = 1e-12

COMPANY_COLUMN = "company"
COMMUNITY_COLUMN = "community"
ELIGIBILITY_COLUMN = "analysis_eligible"

YEAR_COLUMN_CANDIDATES = [
    "release_year",
    "year",
    "song_year",
    "release_date_year",
    "release_year_clean",
]


# ======================================================================
# UTILITIES
# ======================================================================


def print_heading(title: str) -> None:
    """Print a consistent console heading."""
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def community_column_name(community: int) -> str:
    """Return the canonical profile-column name for a community."""
    return f"community_{community}"


def cosine_distance(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
) -> float:
    """Calculate cosine distance between two non-zero vectors."""
    vector_a = np.asarray(vector_a, dtype=float)
    vector_b = np.asarray(vector_b, dtype=float)

    norm_a = float(np.linalg.norm(vector_a))
    norm_b = float(np.linalg.norm(vector_b))

    if norm_a <= EPSILON or norm_b <= EPSILON:
        return np.nan

    similarity = float(
        np.dot(vector_a, vector_b) / (norm_a * norm_b)
    )

    return float(
        1.0 - np.clip(similarity, -1.0, 1.0)
    )


def safe_slope(
    x_values: pd.Series | np.ndarray,
    y_values: pd.Series | np.ndarray,
) -> float:
    """Return an ordinary least-squares slope or NaN when undefined."""
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 2 or np.unique(x).size < 2:
        return np.nan

    return float(np.polyfit(x, y, 1)[0])


def safe_correlation(
    x_values: pd.Series | np.ndarray,
    y_values: pd.Series | np.ndarray,
) -> float:
    """Return Pearson correlation or NaN when undefined."""
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if (
        len(x) < 2
        or np.std(x) <= EPSILON
        or np.std(y) <= EPSILON
    ):
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def classify_direction(value: float, tolerance: float = 1e-6) -> str:
    """Convert a signed statistic to increasing/decreasing/stable."""
    if not np.isfinite(value):
        return "insufficient_data"
    if value > tolerance:
        return "increasing"
    if value < -tolerance:
        return "decreasing"
    return "stable"


def classify_agreement(probability: float) -> str:
    """Classify cross-window agreement probability."""
    if not np.isfinite(probability):
        return "insufficient_data"
    if probability >= 0.90:
        return "highly_robust"
    if probability >= 0.75:
        return "robust"
    if probability >= 0.50:
        return "moderately_robust"
    return "window_sensitive"


# ======================================================================
# DATA VALIDATION AND PREPARATION
# ======================================================================


def identify_year_column(dataframe: pd.DataFrame) -> str:
    """Identify the release-year column from supported candidates."""
    for column in YEAR_COLUMN_CANDIDATES:
        if column in dataframe.columns:
            return column

    raise ValueError(
        "No supported release-year column was found. Expected one of: "
        + ", ".join(YEAR_COLUMN_CANDIDATES)
    )


def validate_master_dataset(
    dataframe: pd.DataFrame,
    year_column: str,
) -> None:
    """Validate required columns in the master dataset."""
    required_columns = {
        COMPANY_COLUMN,
        COMMUNITY_COLUMN,
        ELIGIBILITY_COLUMN,
        year_column,
    }

    missing_columns = sorted(
        required_columns.difference(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "The master dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    if dataframe.empty:
        raise ValueError("The master dataset is empty.")


def prepare_master_dataset(
    dataframe: pd.DataFrame,
    year_column: str,
) -> pd.DataFrame:
    """Filter eligible rows and standardise analysis fields."""
    working = dataframe.copy()

    eligibility_values = working[ELIGIBILITY_COLUMN]

    if pd.api.types.is_bool_dtype(eligibility_values):
        eligible_mask = eligibility_values.fillna(False)
    else:
        eligible_mask = (
            eligibility_values.astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1", "yes"})
        )

    working = working.loc[eligible_mask].copy()

    working[COMPANY_COLUMN] = (
        working[COMPANY_COLUMN]
        .astype(str)
        .str.strip()
    )

    working[COMMUNITY_COLUMN] = pd.to_numeric(
        working[COMMUNITY_COLUMN],
        errors="coerce",
    )

    working[year_column] = pd.to_numeric(
        working[year_column],
        errors="coerce",
    )

    working = working.dropna(
        subset=[
            COMPANY_COLUMN,
            COMMUNITY_COLUMN,
            year_column,
        ]
    ).copy()

    working[COMMUNITY_COLUMN] = working[
        COMMUNITY_COLUMN
    ].astype(int)

    working[year_column] = working[year_column].astype(int)

    working = working.loc[
        working[COMPANY_COLUMN].ne("")
    ].reset_index(drop=True)

    if working.empty:
        raise ValueError(
            "No valid analysis-eligible rows remain after preparation."
        )

    return working


# ======================================================================
# WINDOW CONSTRUCTION
# ======================================================================


def build_window_definitions(
    minimum_year: int,
    maximum_year: int,
) -> pd.DataFrame:
    """Build annual and rolling temporal-window definitions."""
    rows = []

    for window_name, window_width in WINDOW_SPECS.items():
        for end_year in range(minimum_year, maximum_year + 1):
            start_year = end_year - window_width + 1

            if start_year < minimum_year:
                continue

            rows.append(
                {
                    "window_name": window_name,
                    "window_width": int(window_width),
                    "window_start": int(start_year),
                    "window_end": int(end_year),
                    "window_midpoint": float(
                        (start_year + end_year) / 2.0
                    ),
                    "window_label": (
                        str(end_year)
                        if window_width == 1
                        else f"{start_year}-{end_year}"
                    ),
                }
            )

    return pd.DataFrame(rows)


# ======================================================================
# WINDOW-LEVEL PROFILE CALCULATION
# ======================================================================


def calculate_profile_row(
    company_data: pd.DataFrame,
    company: str,
    communities: list[int],
    window_definition: pd.Series,
) -> dict:
    """Calculate one company profile in one temporal window."""
    n_songs = len(company_data)
    community_counts = (
        company_data[COMMUNITY_COLUMN].value_counts()
    )

    row = {
        "window_name": window_definition["window_name"],
        "window_width": int(window_definition["window_width"]),
        "window_start": int(window_definition["window_start"]),
        "window_end": int(window_definition["window_end"]),
        "window_midpoint": float(
            window_definition["window_midpoint"]
        ),
        "window_label": window_definition["window_label"],
        "company": company,
        "n_songs": int(n_songs),
        "meets_minimum_sample": bool(
            n_songs >= MIN_SONGS_PER_COMPANY_WINDOW
        ),
    }

    for community in communities:
        row[community_column_name(community)] = float(
            community_counts.get(community, 0) / n_songs
        )

    fingerprint = np.array(
        [
            row[community_column_name(community)]
            for community in communities
        ],
        dtype=float,
    )

    positive = fingerprint[fingerprint > 0]
    entropy = float(-np.sum(positive * np.log(positive)))
    dominant_index = int(np.argmax(fingerprint))

    row["entropy"] = entropy
    row["effective_communities"] = float(np.exp(entropy))
    row["dominant_community"] = int(
        communities[dominant_index]
    )
    row["dominant_community_share"] = float(
        fingerprint[dominant_index]
    )

    return row


def build_window_profiles(
    master: pd.DataFrame,
    year_column: str,
    companies: list[str],
    communities: list[int],
    windows: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate company profiles for every temporal window."""
    rows = []

    for _, window in windows.iterrows():
        window_data = master.loc[
            master[year_column].between(
                int(window["window_start"]),
                int(window["window_end"]),
                inclusive="both",
            )
        ]

        for company in companies:
            company_data = window_data.loc[
                window_data[COMPANY_COLUMN] == company
            ]

            if company_data.empty:
                continue

            rows.append(
                calculate_profile_row(
                    company_data=company_data,
                    company=company,
                    communities=communities,
                    window_definition=window,
                )
            )

    profiles = pd.DataFrame(rows)

    if profiles.empty:
        raise ValueError(
            "No temporal-window profiles could be constructed."
        )

    return profiles


# ======================================================================
# WINDOW PAIRS AND RANKINGS
# ======================================================================


def build_window_pairs(
    profiles: pd.DataFrame,
    communities: list[int],
) -> pd.DataFrame:
    """Calculate company-pair distances in every valid window."""
    community_columns = [
        community_column_name(community)
        for community in communities
    ]

    rows = []
    group_columns = [
        "window_name",
        "window_width",
        "window_start",
        "window_end",
        "window_midpoint",
        "window_label",
    ]

    for keys, group in profiles.groupby(
        group_columns,
        sort=True,
    ):
        valid = group.loc[group["meets_minimum_sample"]]

        if valid["company"].nunique() < MIN_COMPANIES_PER_WINDOW:
            continue

        profile_lookup = {
            row["company"]: row[
                community_columns
            ].to_numpy(dtype=float)
            for _, row in valid.iterrows()
        }

        metadata = dict(zip(group_columns, keys))

        for company_a, company_b in combinations(
            sorted(profile_lookup),
            2,
        ):
            rows.append(
                {
                    **metadata,
                    "company_a": company_a,
                    "company_b": company_b,
                    "company_pair": f"{company_a}|{company_b}",
                    "cosine_distance": cosine_distance(
                        profile_lookup[company_a],
                        profile_lookup[company_b],
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_window_rankings(
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Rank companies by mean pairwise distance in each window."""
    if pairs.empty:
        return pd.DataFrame()

    rows = []
    group_columns = [
        "window_name",
        "window_width",
        "window_start",
        "window_end",
        "window_midpoint",
        "window_label",
    ]

    for keys, group in pairs.groupby(
        group_columns,
        sort=True,
    ):
        companies = sorted(
            set(group["company_a"]).union(
                set(group["company_b"])
            )
        )

        metadata = dict(zip(group_columns, keys))
        temporary_rows = []

        for company in companies:
            distances = group.loc[
                (group["company_a"] == company)
                | (group["company_b"] == company),
                "cosine_distance",
            ].dropna()

            if distances.empty:
                continue

            temporary_rows.append(
                {
                    **metadata,
                    "company": company,
                    "n_pairwise_distances": int(len(distances)),
                    "mean_pairwise_distance": float(
                        distances.mean()
                    ),
                    "median_pairwise_distance": float(
                        distances.median()
                    ),
                    "maximum_pairwise_distance": float(
                        distances.max()
                    ),
                }
            )

        temporary = pd.DataFrame(temporary_rows)

        if temporary.empty:
            continue

        temporary["distinctiveness_rank"] = (
            temporary["mean_pairwise_distance"]
            .rank(method="min", ascending=False)
            .astype(int)
        )

        rows.extend(temporary.to_dict(orient="records"))

    return pd.DataFrame(rows)


# ======================================================================
# DRIFT AND CONVERGENCE
# ======================================================================


def build_temporal_drift(
    profiles: pd.DataFrame,
    communities: list[int],
) -> pd.DataFrame:
    """Calculate consecutive-window profile drift for each company."""
    community_columns = [
        community_column_name(community)
        for community in communities
    ]

    rows = []

    for (window_name, company), group in profiles.groupby(
        ["window_name", "company"],
        sort=True,
    ):
        valid = (
            group.loc[group["meets_minimum_sample"]]
            .sort_values("window_end")
            .reset_index(drop=True)
        )

        for position in range(1, len(valid)):
            previous = valid.iloc[position - 1]
            current = valid.iloc[position]

            previous_vector = previous[
                community_columns
            ].to_numpy(dtype=float)
            current_vector = current[
                community_columns
            ].to_numpy(dtype=float)

            rows.append(
                {
                    "window_name": window_name,
                    "window_width": int(current["window_width"]),
                    "company": company,
                    "previous_window_start": int(
                        previous["window_start"]
                    ),
                    "previous_window_end": int(
                        previous["window_end"]
                    ),
                    "current_window_start": int(
                        current["window_start"]
                    ),
                    "current_window_end": int(
                        current["window_end"]
                    ),
                    "current_window_midpoint": float(
                        current["window_midpoint"]
                    ),
                    "profile_cosine_drift": cosine_distance(
                        previous_vector,
                        current_vector,
                    ),
                    "profile_l1_drift": float(
                        np.sum(
                            np.abs(
                                current_vector - previous_vector
                            )
                        )
                    ),
                    "entropy_change": float(
                        current["entropy"] - previous["entropy"]
                    ),
                    "dominant_community_changed": bool(
                        current["dominant_community"]
                        != previous["dominant_community"]
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_temporal_convergence(
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise whether company pairs converge or diverge over time."""
    rows = []

    if pairs.empty:
        return pd.DataFrame(rows)

    for (window_name, company_pair), group in pairs.groupby(
        ["window_name", "company_pair"],
        sort=True,
    ):
        ordered = group.sort_values("window_end")
        slope = safe_slope(
            ordered["window_midpoint"],
            ordered["cosine_distance"],
        )

        rows.append(
            {
                "window_name": window_name,
                "window_width": int(
                    ordered["window_width"].iloc[0]
                ),
                "company_pair": company_pair,
                "company_a": ordered["company_a"].iloc[0],
                "company_b": ordered["company_b"].iloc[0],
                "n_windows": int(len(ordered)),
                "first_window_end": int(
                    ordered["window_end"].iloc[0]
                ),
                "last_window_end": int(
                    ordered["window_end"].iloc[-1]
                ),
                "first_distance": float(
                    ordered["cosine_distance"].iloc[0]
                ),
                "last_distance": float(
                    ordered["cosine_distance"].iloc[-1]
                ),
                "net_distance_change": float(
                    ordered["cosine_distance"].iloc[-1]
                    - ordered["cosine_distance"].iloc[0]
                ),
                "distance_slope": slope,
                "temporal_direction": (
                    "diverging"
                    if classify_direction(slope) == "increasing"
                    else "converging"
                    if classify_direction(slope) == "decreasing"
                    else classify_direction(slope)
                ),
            }
        )

    return pd.DataFrame(rows)


# ======================================================================
# CROSS-WINDOW SENSITIVITY SUMMARIES
# ======================================================================


def build_window_summary(
    profiles: pd.DataFrame,
    rankings: pd.DataFrame,
    drift: pd.DataFrame,
    convergence: pd.DataFrame,
) -> pd.DataFrame:
    """Build one summary row per temporal-window definition."""
    rows = []

    for window_name, width in WINDOW_SPECS.items():
        profile_data = profiles.loc[
            profiles["window_name"] == window_name
        ]
        ranking_data = rankings.loc[
            rankings["window_name"] == window_name
        ] if not rankings.empty else pd.DataFrame()
        drift_data = drift.loc[
            drift["window_name"] == window_name
        ] if not drift.empty else pd.DataFrame()
        convergence_data = convergence.loc[
            convergence["window_name"] == window_name
        ] if not convergence.empty else pd.DataFrame()

        valid_profiles = profile_data.loc[
            profile_data["meets_minimum_sample"]
        ]

        rows.append(
            {
                "window_name": window_name,
                "window_width": int(width),
                "n_defined_windows": int(
                    profile_data[
                        ["window_start", "window_end"]
                    ].drop_duplicates().shape[0]
                ),
                "n_valid_company_window_profiles": int(
                    len(valid_profiles)
                ),
                "minimum_company_window_n": (
                    int(valid_profiles["n_songs"].min())
                    if not valid_profiles.empty
                    else 0
                ),
                "median_company_window_n": (
                    float(valid_profiles["n_songs"].median())
                    if not valid_profiles.empty
                    else np.nan
                ),
                "mean_entropy": (
                    float(valid_profiles["entropy"].mean())
                    if not valid_profiles.empty
                    else np.nan
                ),
                "mean_consecutive_profile_drift": (
                    float(
                        drift_data[
                            "profile_cosine_drift"
                        ].mean()
                    )
                    if not drift_data.empty
                    else np.nan
                ),
                "median_consecutive_profile_drift": (
                    float(
                        drift_data[
                            "profile_cosine_drift"
                        ].median()
                    )
                    if not drift_data.empty
                    else np.nan
                ),
                "dominant_community_change_probability": (
                    float(
                        drift_data[
                            "dominant_community_changed"
                        ].mean()
                    )
                    if not drift_data.empty
                    else np.nan
                ),
                "n_ranking_windows": int(
                    ranking_data[
                        ["window_start", "window_end"]
                    ].drop_duplicates().shape[0]
                ) if not ranking_data.empty else 0,
                "mean_pair_distance_slope": (
                    float(
                        convergence_data[
                            "distance_slope"
                        ].mean()
                    )
                    if not convergence_data.empty
                    else np.nan
                ),
                "converging_pair_fraction": (
                    float(
                        (
                            convergence_data[
                                "temporal_direction"
                            ] == "converging"
                        ).mean()
                    )
                    if not convergence_data.empty
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def build_company_sensitivity_summary(
    profiles: pd.DataFrame,
    rankings: pd.DataFrame,
    drift: pd.DataFrame,
) -> pd.DataFrame:
    """Compare company-level temporal conclusions across windows."""
    companies = sorted(profiles["company"].unique())
    rows = []

    for company in companies:
        direction_rows = []
        rank_rows = []
        drift_rows = []

        for window_name in WINDOW_SPECS:
            company_profiles = (
                profiles.loc[
                    (profiles["company"] == company)
                    & (profiles["window_name"] == window_name)
                    & profiles["meets_minimum_sample"]
                ]
                .sort_values("window_end")
            )

            entropy_slope = safe_slope(
                company_profiles["window_midpoint"],
                company_profiles["entropy"],
            )

            direction_rows.append(
                {
                    "window_name": window_name,
                    "entropy_slope": entropy_slope,
                    "entropy_direction": classify_direction(
                        entropy_slope
                    ),
                }
            )

            if not rankings.empty:
                company_rankings = rankings.loc[
                    (rankings["company"] == company)
                    & (rankings["window_name"] == window_name)
                ]

                rank_rows.append(
                    {
                        "window_name": window_name,
                        "mean_rank": (
                            float(
                                company_rankings[
                                    "distinctiveness_rank"
                                ].mean()
                            )
                            if not company_rankings.empty
                            else np.nan
                        ),
                        "rank_slope": safe_slope(
                            company_rankings[
                                "window_midpoint"
                            ],
                            company_rankings[
                                "distinctiveness_rank"
                            ],
                        ),
                    }
                )

            if not drift.empty:
                company_drift = drift.loc[
                    (drift["company"] == company)
                    & (drift["window_name"] == window_name)
                ]

                drift_rows.append(
                    {
                        "window_name": window_name,
                        "mean_profile_drift": (
                            float(
                                company_drift[
                                    "profile_cosine_drift"
                                ].mean()
                            )
                            if not company_drift.empty
                            else np.nan
                        ),
                    }
                )

        direction_frame = pd.DataFrame(direction_rows)
        usable_directions = direction_frame.loc[
            direction_frame["entropy_direction"]
            != "insufficient_data"
        ]

        if usable_directions.empty:
            dominant_direction = "insufficient_data"
            direction_agreement = np.nan
        else:
            counts = usable_directions[
                "entropy_direction"
            ].value_counts()
            dominant_direction = str(counts.index[0])
            direction_agreement = float(
                counts.iloc[0] / counts.sum()
            )

        rank_frame = pd.DataFrame(rank_rows)
        drift_frame = pd.DataFrame(drift_rows)

        mean_ranks = rank_frame[
            "mean_rank"
        ].dropna() if not rank_frame.empty else pd.Series(dtype=float)

        mean_drifts = drift_frame[
            "mean_profile_drift"
        ].dropna() if not drift_frame.empty else pd.Series(dtype=float)

        rows.append(
            {
                "company": company,
                "n_window_definitions": int(
                    len(WINDOW_SPECS)
                ),
                "dominant_entropy_direction": dominant_direction,
                "entropy_direction_agreement": direction_agreement,
                "entropy_robustness": classify_agreement(
                    direction_agreement
                ),
                "mean_rank_across_definitions": (
                    float(mean_ranks.mean())
                    if not mean_ranks.empty
                    else np.nan
                ),
                "mean_rank_range_across_definitions": (
                    float(mean_ranks.max() - mean_ranks.min())
                    if len(mean_ranks) >= 2
                    else np.nan
                ),
                "mean_drift_across_definitions": (
                    float(mean_drifts.mean())
                    if not mean_drifts.empty
                    else np.nan
                ),
                "drift_range_across_definitions": (
                    float(mean_drifts.max() - mean_drifts.min())
                    if len(mean_drifts) >= 2
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def build_pair_sensitivity_summary(
    convergence: pd.DataFrame,
) -> pd.DataFrame:
    """Compare convergence directions across window definitions."""
    if convergence.empty:
        return pd.DataFrame()

    rows = []

    for company_pair, group in convergence.groupby(
        "company_pair",
        sort=True,
    ):
        usable = group.loc[
            ~group["temporal_direction"].isin(
                ["insufficient_data"]
            )
        ]

        if usable.empty:
            dominant_direction = "insufficient_data"
            direction_agreement = np.nan
        else:
            counts = usable[
                "temporal_direction"
            ].value_counts()
            dominant_direction = str(counts.index[0])
            direction_agreement = float(
                counts.iloc[0] / counts.sum()
            )

        rows.append(
            {
                "company_pair": company_pair,
                "company_a": group["company_a"].iloc[0],
                "company_b": group["company_b"].iloc[0],
                "n_window_definitions": int(len(group)),
                "dominant_temporal_direction": dominant_direction,
                "direction_agreement": direction_agreement,
                "direction_robustness": classify_agreement(
                    direction_agreement
                ),
                "mean_distance_slope": float(
                    group["distance_slope"].mean()
                ),
                "minimum_distance_slope": float(
                    group["distance_slope"].min()
                ),
                "maximum_distance_slope": float(
                    group["distance_slope"].max()
                ),
                "slope_range": float(
                    group["distance_slope"].max()
                    - group["distance_slope"].min()
                ),
            }
        )

    return pd.DataFrame(rows)


def build_publication_summary(
    window_summary: pd.DataFrame,
    company_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build a compact reviewer-facing temporal robustness summary."""
    rows = []

    for _, row in window_summary.iterrows():
        rows.append(
            {
                "analysis": "Temporal window definition",
                "entity": row["window_name"],
                "primary_statistic": "Mean consecutive profile drift",
                "primary_value": row[
                    "mean_consecutive_profile_drift"
                ],
                "secondary_statistic": (
                    "Dominant-community change probability"
                ),
                "secondary_value": row[
                    "dominant_community_change_probability"
                ],
                "agreement": np.nan,
                "largest_deviation": np.nan,
                "conclusion": (
                    "descriptive_window_result"
                ),
            }
        )

    for _, row in company_summary.iterrows():
        rows.append(
            {
                "analysis": "Company temporal sensitivity",
                "entity": row["company"],
                "primary_statistic": (
                    "Entropy-direction agreement"
                ),
                "primary_value": row[
                    "entropy_direction_agreement"
                ],
                "secondary_statistic": (
                    "Mean rank across definitions"
                ),
                "secondary_value": row[
                    "mean_rank_across_definitions"
                ],
                "agreement": row[
                    "dominant_entropy_direction"
                ],
                "largest_deviation": row[
                    "drift_range_across_definitions"
                ],
                "conclusion": row["entropy_robustness"],
            }
        )

    if not pair_summary.empty:
        for _, row in pair_summary.iterrows():
            rows.append(
                {
                    "analysis": "Pairwise temporal sensitivity",
                    "entity": row["company_pair"],
                    "primary_statistic": (
                        "Convergence-direction agreement"
                    ),
                    "primary_value": row[
                        "direction_agreement"
                    ],
                    "secondary_statistic": (
                        "Mean distance slope"
                    ),
                    "secondary_value": row[
                        "mean_distance_slope"
                    ],
                    "agreement": row[
                        "dominant_temporal_direction"
                    ],
                    "largest_deviation": row["slope_range"],
                    "conclusion": row[
                        "direction_robustness"
                    ],
                }
            )

    return pd.DataFrame(rows)


# ======================================================================
# CONSOLE OUTPUT
# ======================================================================


def print_results(
    master: pd.DataFrame,
    year_column: str,
    profiles: pd.DataFrame,
    window_summary: pd.DataFrame,
    company_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
) -> None:
    """Print principal temporal-window sensitivity results."""
    print_heading("WP6e : TEMPORAL WINDOW SENSITIVITY")

    print(f"Master dataset          : {MASTER_PATH}")
    print(f"Eligible rows           : {len(master)}")
    print(f"Year column             : {year_column}")
    print(
        "Observed year range     : "
        f"{master[year_column].min()}-"
        f"{master[year_column].max()}"
    )
    print(
        "Window definitions      : "
        + ", ".join(WINDOW_SPECS)
    )
    print(
        "Minimum songs/company   : "
        f"{MIN_SONGS_PER_COMPANY_WINDOW}"
    )
    print(f"Profile rows generated  : {len(profiles)}")

    print_heading("WINDOW-LEVEL SUMMARY")
    print(window_summary.to_string(index=False))

    print_heading("COMPANY TEMPORAL SENSITIVITY")
    print(company_summary.to_string(index=False))

    print_heading("PAIRWISE CONVERGENCE SENSITIVITY")
    if pair_summary.empty:
        print("No valid pairwise temporal summaries were produced.")
    else:
        print(pair_summary.to_string(index=False))


# ======================================================================
# MAIN
# ======================================================================


def main() -> None:
    """Execute WP6e temporal-window sensitivity analysis."""
    print_heading("WP6e : TEMPORAL WINDOW SENSITIVITY")

    if not MASTER_PATH.exists():
        raise FileNotFoundError(
            "The authoritative master dataset does not exist: "
            f"{MASTER_PATH}"
        )

    master_raw = pd.read_parquet(MASTER_PATH)
    year_column = identify_year_column(master_raw)

    validate_master_dataset(
        dataframe=master_raw,
        year_column=year_column,
    )

    master = prepare_master_dataset(
        dataframe=master_raw,
        year_column=year_column,
    )

    companies = sorted(
        master[COMPANY_COLUMN].unique().tolist()
    )
    communities = sorted(
        master[COMMUNITY_COLUMN].unique().tolist()
    )

    minimum_year = int(master[year_column].min())
    maximum_year = int(master[year_column].max())

    windows = build_window_definitions(
        minimum_year=minimum_year,
        maximum_year=maximum_year,
    )

    print(f"Master dataset          : {MASTER_PATH}")
    print(f"Master rows loaded      : {len(master_raw)}")
    print(f"Eligible rows retained  : {len(master)}")
    print(f"Year column             : {year_column}")
    print(f"Companies analysed      : {len(companies)}")
    print(f"Communities analysed    : {len(communities)}")
    print(
        "Observed year range     : "
        f"{minimum_year}-{maximum_year}"
    )
    print(f"Window rows defined     : {len(windows)}")

    profiles = build_window_profiles(
        master=master,
        year_column=year_column,
        companies=companies,
        communities=communities,
        windows=windows,
    )

    pairs = build_window_pairs(
        profiles=profiles,
        communities=communities,
    )

    rankings = build_window_rankings(pairs=pairs)

    drift = build_temporal_drift(
        profiles=profiles,
        communities=communities,
    )

    convergence = build_temporal_convergence(
        pairs=pairs
    )

    window_summary = build_window_summary(
        profiles=profiles,
        rankings=rankings,
        drift=drift,
        convergence=convergence,
    )

    company_summary = build_company_sensitivity_summary(
        profiles=profiles,
        rankings=rankings,
        drift=drift,
    )

    pair_summary = build_pair_sensitivity_summary(
        convergence=convergence
    )

    publication_summary = build_publication_summary(
        window_summary=window_summary,
        company_summary=company_summary,
        pair_summary=pair_summary,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    profiles.to_csv(WINDOW_PROFILES_PATH, index=False)
    pairs.to_csv(WINDOW_PAIRS_PATH, index=False)
    rankings.to_csv(WINDOW_RANKINGS_PATH, index=False)
    drift.to_csv(WINDOW_DRIFT_PATH, index=False)
    convergence.to_csv(WINDOW_CONVERGENCE_PATH, index=False)
    window_summary.to_csv(WINDOW_SUMMARY_PATH, index=False)
    company_summary.to_csv(COMPANY_SUMMARY_PATH, index=False)
    pair_summary.to_csv(PAIR_SUMMARY_PATH, index=False)
    publication_summary.to_csv(
        PUBLICATION_SUMMARY_PATH,
        index=False,
    )

    print_results(
        master=master,
        year_column=year_column,
        profiles=profiles,
        window_summary=window_summary,
        company_summary=company_summary,
        pair_summary=pair_summary,
    )

    print_heading("FILES WRITTEN")
    for output_path in [
        WINDOW_PROFILES_PATH,
        WINDOW_PAIRS_PATH,
        WINDOW_RANKINGS_PATH,
        WINDOW_DRIFT_PATH,
        WINDOW_CONVERGENCE_PATH,
        WINDOW_SUMMARY_PATH,
        COMPANY_SUMMARY_PATH,
        PAIR_SUMMARY_PATH,
        PUBLICATION_SUMMARY_PATH,
    ]:
        print(output_path)

    print("=" * 78)


if __name__ == "__main__":
    main()
