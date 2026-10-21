"""
WP6c: Leave-one-artist-out robustness of company semantic profiles.

This script tests whether the company-level semantic findings are
disproportionately driven by individual artists.

For every eligible artist, the script:

1. Removes all songs associated with that artist.
2. Reconstructs company community fingerprints.
3. Recalculates pairwise cosine distances.
4. Recalculates company distinctiveness rankings.
5. Measures changes relative to the complete observed dataset.

The analysis uses the authoritative master dataset:

    outputs/tables/company_master_analysis_v1.parquet

The script produces raw leave-one-artist-out results, artist-influence
statistics, company and pairwise summaries, global ranking-stability
statistics, and a compact publication-facing summary.
"""

from __future__ import annotations

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
    / "leave_one_artist_out"
)

LOAO_PROFILES_PATH = (
    OUTPUT_DIR
    / "leave_one_artist_out_profiles.csv"
)

LOAO_PAIRS_PATH = (
    OUTPUT_DIR
    / "leave_one_artist_out_pairs.csv"
)

LOAO_RANKINGS_PATH = (
    OUTPUT_DIR
    / "leave_one_artist_out_rankings.csv"
)

PROFILE_CHANGES_PATH = (
    OUTPUT_DIR
    / "leave_one_artist_out_profile_changes.csv"
)

PAIR_CHANGES_PATH = (
    OUTPUT_DIR
    / "leave_one_artist_out_pair_changes.csv"
)

RANKING_CHANGES_PATH = (
    OUTPUT_DIR
    / "leave_one_artist_out_ranking_changes.csv"
)

ARTIST_INFLUENCE_PATH = (
    OUTPUT_DIR
    / "leave_one_artist_out_artist_influence.csv"
)

COMPANY_SUMMARY_PATH = (
    OUTPUT_DIR
    / "leave_one_artist_out_company_summary.csv"
)

PAIR_SUMMARY_PATH = (
    OUTPUT_DIR
    / "leave_one_artist_out_pair_summary.csv"
)

RANKING_STABILITY_PATH = (
    OUTPUT_DIR
    / "leave_one_artist_out_ranking_stability.csv"
)

PUBLICATION_SUMMARY_PATH = (
    OUTPUT_DIR
    / "leave_one_artist_out_summary.csv"
)

COMPANY_COLUMN = "company"
COMMUNITY_COLUMN = "community"
ELIGIBILITY_COLUMN = "analysis_eligible"

ARTIST_COLUMN_CANDIDATES = [
    "artist",
    "artist_name",
    "artist_clean",
    "artist_standardised",
    "artist_standardized",
    "primary_artist",
    "group",
    "group_name",
]

COMPANY_METRIC_COLUMNS = [
    "degree",
    "weighted_degree",
    "bridge_affinity_score",
    "local_novelty",
    "centroid_novelty",
]

EPSILON = 1e-12

PROFILE_SHIFT_SMALL = 0.025
PROFILE_SHIFT_MODERATE = 0.050

PAIR_SHIFT_SMALL = 0.010
PAIR_SHIFT_MODERATE = 0.025

DISTINCTIVENESS_SHIFT_SMALL = 0.010
DISTINCTIVENESS_SHIFT_MODERATE = 0.025


# ======================================================================
# GENERAL UTILITIES
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


def community_column_name(
    community: int,
) -> str:
    """
    Return the canonical profile-column name for a community.
    """

    return f"community_{community}"


def sanitise_text_column(
    series: pd.Series,
) -> pd.Series:
    """
    Standardise a text column and convert empty values to missing.
    """

    cleaned = (
        series
        .astype("string")
        .str.strip()
    )

    cleaned = cleaned.mask(
        cleaned.isin(
            [
                "",
                "nan",
                "None",
                "<NA>",
            ]
        )
    )

    return cleaned


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
# DATASET VALIDATION
# ======================================================================

def detect_artist_column(
    dataframe: pd.DataFrame,
) -> str:
    """
    Detect the artist column from a list of expected alternatives.
    """

    for candidate in ARTIST_COLUMN_CANDIDATES:
        if candidate in dataframe.columns:
            return candidate

    artist_like_columns = [
        column
        for column in dataframe.columns
        if "artist" in column.lower()
    ]

    if len(
        artist_like_columns
    ) == 1:
        return artist_like_columns[
            0
        ]

    if artist_like_columns:
        raise ValueError(
            "Multiple possible artist columns were detected: "
            + ", ".join(
                artist_like_columns
            )
            + ". Add the correct column name to "
            "ARTIST_COLUMN_CANDIDATES."
        )

    raise ValueError(
        "No artist column could be detected. Available columns are: "
        + ", ".join(
            dataframe.columns.astype(str)
        )
    )


def validate_master_dataset(
    dataframe: pd.DataFrame,
    artist_column: str,
) -> None:
    """
    Validate the authoritative master analysis dataset.
    """

    required_columns = {
        COMPANY_COLUMN,
        COMMUNITY_COLUMN,
        ELIGIBILITY_COLUMN,
        artist_column,
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


def prepare_master_dataset(
    dataframe: pd.DataFrame,
    artist_column: str,
) -> pd.DataFrame:
    """
    Filter eligible rows and standardise analysis columns.
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
    ] = sanitise_text_column(
        working[
            COMPANY_COLUMN
        ]
    )

    working[
        artist_column
    ] = sanitise_text_column(
        working[
            artist_column
        ]
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

    missing_company_rows = int(
        working[
            COMPANY_COLUMN
        ].isna().sum()
    )

    missing_artist_rows = int(
        working[
            artist_column
        ].isna().sum()
    )

    if missing_company_rows:
        raise ValueError(
            f"{missing_company_rows} eligible rows have no company."
        )

    if missing_artist_rows:
        print(
            "Rows removed without artist information: "
            f"{missing_artist_rows}"
        )

        working = (
            working.loc[
                working[
                    artist_column
                ].notna()
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

    if working.empty:
        raise ValueError(
            "No eligible rows remain after removing missing artists."
        )

    return working


def identify_communities(
    dataframe: pd.DataFrame,
) -> list[int]:
    """
    Return all observed community labels in numerical order.
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


def validate_artist_company_mapping(
    dataframe: pd.DataFrame,
    artist_column: str,
) -> pd.DataFrame:
    """
    Inspect whether any artist appears under multiple companies.

    The analysis permits this situation, but the mapping is saved in the
    artist-influence output and explicitly reported.
    """

    mapping = (
        dataframe.groupby(
            artist_column,
            as_index=False,
        )[
            COMPANY_COLUMN
        ]
        .agg(
            lambda values: "|".join(
                sorted(
                    values.astype(str).unique()
                )
            )
        )
        .rename(
            columns={
                COMPANY_COLUMN: "companies",
            }
        )
    )

    company_counts = (
        dataframe.groupby(
            artist_column
        )[
            COMPANY_COLUMN
        ]
        .nunique()
        .rename(
            "n_companies"
        )
        .reset_index()
    )

    mapping = mapping.merge(
        company_counts,
        on=artist_column,
        how="left",
        validate="one_to_one",
    )

    return mapping


# ======================================================================
# PROFILE CALCULATION
# ======================================================================

def calculate_company_profile(
    company_dataframe: pd.DataFrame,
    company: str,
    communities: list[int],
    scenario_id: int,
    removed_artist: str,
    is_observed: bool,
) -> dict:
    """
    Construct a company fingerprint and descriptive statistics.
    """

    n_songs = len(
        company_dataframe
    )

    if n_songs == 0:
        raise ValueError(
            f"Company '{company}' has no rows in scenario "
            f"'{removed_artist}'."
        )

    community_counts = (
        company_dataframe[
            COMMUNITY_COLUMN
        ]
        .value_counts()
    )

    row = {
        "scenario_id": int(
            scenario_id
        ),
        "removed_artist": removed_artist,
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


def calculate_profiles_for_scenario(
    sample: pd.DataFrame,
    companies: list[str],
    communities: list[int],
    scenario_id: int,
    removed_artist: str,
    is_observed: bool,
) -> pd.DataFrame:
    """
    Calculate all company profiles for one leave-one-out scenario.
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
                scenario_id=scenario_id,
                removed_artist=removed_artist,
                is_observed=is_observed,
            )
        )

    return pd.DataFrame(
        rows
    )


# ======================================================================
# PAIRWISE DISTANCES AND RANKINGS
# ======================================================================

def calculate_company_pairs(
    profiles: pd.DataFrame,
    communities: list[int],
) -> pd.DataFrame:
    """
    Calculate all pairwise company-profile cosine distances.
    """

    community_columns = [
        community_column_name(
            community
        )
        for community in communities
    ]

    scenario_ids = profiles[
        "scenario_id"
    ].unique()

    if len(
        scenario_ids
    ) != 1:
        raise ValueError(
            "Profiles must belong to one scenario."
        )

    scenario_id = int(
        scenario_ids[
            0
        ]
    )

    removed_artists = profiles[
        "removed_artist"
    ].unique()

    if len(
        removed_artists
    ) != 1:
        raise ValueError(
            "Profiles contain inconsistent removed artists."
        )

    removed_artist = str(
        removed_artists[
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
            "Profiles contain inconsistent observed indicators."
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
        rows.append(
            {
                "scenario_id": scenario_id,
                "removed_artist": removed_artist,
                "is_observed": is_observed,
                "company_a": company_a,
                "company_b": company_b,
                "company_pair": (
                    f"{company_a}|{company_b}"
                ),
                "cosine_distance": cosine_distance(
                    profile_lookup[
                        company_a
                    ],
                    profile_lookup[
                        company_b
                    ],
                ),
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

    scenario_ids = pairs[
        "scenario_id"
    ].unique()

    if len(
        scenario_ids
    ) != 1:
        raise ValueError(
            "Pair rows must belong to one scenario."
        )

    scenario_id = int(
        scenario_ids[
            0
        ]
    )

    removed_artists = pairs[
        "removed_artist"
    ].unique()

    if len(
        removed_artists
    ) != 1:
        raise ValueError(
            "Pair rows contain inconsistent removed artists."
        )

    removed_artist = str(
        removed_artists[
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
            "Pair rows contain inconsistent observed indicators."
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

        expected_count = (
            len(
                companies
            )
            - 1
        )

        if len(
            company_distances
        ) != expected_count:
            raise ValueError(
                f"Unexpected pair count for company '{company}'."
            )

        rows.append(
            {
                "scenario_id": scenario_id,
                "removed_artist": removed_artist,
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

    return (
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


# ======================================================================
# LEAVE-ONE-ARTIST-OUT EXECUTION
# ======================================================================

def run_leave_one_artist_out(
    master: pd.DataFrame,
    artist_column: str,
    companies: list[str],
    communities: list[int],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Run the observed analysis and all leave-one-artist-out scenarios.
    """

    profile_frames = []
    pair_frames = []
    ranking_frames = []

    observed_profiles = (
        calculate_profiles_for_scenario(
            sample=master,
            companies=companies,
            communities=communities,
            scenario_id=0,
            removed_artist="__OBSERVED__",
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

    artists = sorted(
        master[
            artist_column
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    n_artists = len(
        artists
    )

    progress_interval = max(
        1,
        n_artists
        // 20,
    )

    for scenario_id, artist in enumerate(
        artists,
        start=1,
    ):
        reduced_sample = (
            master.loc[
                master[
                    artist_column
                ] != artist
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        remaining_companies = set(
            reduced_sample[
                COMPANY_COLUMN
            ].unique()
        )

        missing_companies = sorted(
            set(
                companies
            ).difference(
                remaining_companies
            )
        )

        if missing_companies:
            raise ValueError(
                "Removing artist "
                f"'{artist}' removes all songs from: "
                + ", ".join(
                    missing_companies
                )
                + ". Leave-one-artist-out company profiles cannot "
                "be calculated for this scenario."
            )

        scenario_profiles = (
            calculate_profiles_for_scenario(
                sample=reduced_sample,
                companies=companies,
                communities=communities,
                scenario_id=scenario_id,
                removed_artist=artist,
                is_observed=False,
            )
        )

        scenario_pairs = (
            calculate_company_pairs(
                profiles=scenario_profiles,
                communities=communities,
            )
        )

        scenario_rankings = (
            calculate_company_rankings(
                pairs=scenario_pairs,
                companies=companies,
            )
        )

        profile_frames.append(
            scenario_profiles
        )

        pair_frames.append(
            scenario_pairs
        )

        ranking_frames.append(
            scenario_rankings
        )

        if (
            scenario_id
            % progress_interval
            == 0
            or scenario_id
            == n_artists
        ):
            print(
                "LOAO progress          : "
                f"{scenario_id}/{n_artists}"
            )

    return (
        pd.concat(
            profile_frames,
            ignore_index=True,
        ),
        pd.concat(
            pair_frames,
            ignore_index=True,
        ),
        pd.concat(
            ranking_frames,
            ignore_index=True,
        ),
    )


# ======================================================================
# PROFILE CHANGES
# ======================================================================

def build_profile_changes(
    profiles: pd.DataFrame,
    master: pd.DataFrame,
    artist_column: str,
    communities: list[int],
) -> pd.DataFrame:
    """
    Measure company-profile changes relative to the observed profiles.
    """

    community_columns = [
        community_column_name(
            community
        )
        for community in communities
    ]

    observed = (
        profiles.loc[
            profiles[
                "is_observed"
            ]
        ]
        .set_index(
            "company"
        )
    )

    scenarios = profiles.loc[
        ~profiles[
            "is_observed"
        ]
    ]

    artist_song_counts = (
        master.groupby(
            [
                artist_column,
                COMPANY_COLUMN,
            ]
        )
        .size()
        .rename(
            "removed_artist_song_count"
        )
        .reset_index()
        .rename(
            columns={
                artist_column: "removed_artist",
                COMPANY_COLUMN: "artist_company",
            }
        )
    )

    company_song_counts = (
        master.groupby(
            COMPANY_COLUMN
        )
        .size()
        .rename(
            "observed_company_song_count"
        )
        .reset_index()
        .rename(
            columns={
                COMPANY_COLUMN: "artist_company",
            }
        )
    )

    artist_song_counts = artist_song_counts.merge(
        company_song_counts,
        on="artist_company",
        how="left",
        validate="many_to_one",
    )

    artist_song_counts[
        "removed_company_song_share"
    ] = (
        artist_song_counts[
            "removed_artist_song_count"
        ]
        / artist_song_counts[
            "observed_company_song_count"
        ]
    )

    rows = []

    for _, scenario_row in scenarios.iterrows():
        company = str(
            scenario_row[
                "company"
            ]
        )

        removed_artist = str(
            scenario_row[
                "removed_artist"
            ]
        )

        observed_row = observed.loc[
            company
        ]

        observed_vector = observed_row[
            community_columns
        ].to_numpy(
            dtype=float
        )

        scenario_vector = scenario_row[
            community_columns
        ].to_numpy(
            dtype=float
        )

        difference = (
            scenario_vector
            - observed_vector
        )

        l1_shift = float(
            np.sum(
                np.abs(
                    difference
                )
            )
        )

        l2_shift = float(
            np.linalg.norm(
                difference
            )
        )

        maximum_absolute_shift = float(
            np.max(
                np.abs(
                    difference
                )
            )
        )

        profile_cosine_distance = (
            cosine_distance(
                observed_vector,
                scenario_vector,
            )
        )

        artist_company_rows = (
            artist_song_counts.loc[
                (
                    artist_song_counts[
                        "removed_artist"
                    ] == removed_artist
                )
                &
                (
                    artist_song_counts[
                        "artist_company"
                    ] == company
                )
            ]
        )

        if artist_company_rows.empty:
            removed_song_count = 0
            removed_song_share = 0.0
            affected_company = False
        else:
            count_row = (
                artist_company_rows.iloc[
                    0
                ]
            )

            removed_song_count = int(
                count_row[
                    "removed_artist_song_count"
                ]
            )

            removed_song_share = float(
                count_row[
                    "removed_company_song_share"
                ]
            )

            affected_company = True

        rows.append(
            {
                "scenario_id": int(
                    scenario_row[
                        "scenario_id"
                    ]
                ),
                "removed_artist": removed_artist,
                "company": company,
                "affected_company": affected_company,
                "removed_artist_song_count": removed_song_count,
                "removed_company_song_share": removed_song_share,
                "observed_n_songs": int(
                    observed_row[
                        "n_songs"
                    ]
                ),
                "remaining_n_songs": int(
                    scenario_row[
                        "n_songs"
                    ]
                ),
                "profile_cosine_distance": (
                    profile_cosine_distance
                ),
                "profile_l1_shift": l1_shift,
                "profile_l2_shift": l2_shift,
                "maximum_community_share_shift": (
                    maximum_absolute_shift
                ),
                "observed_entropy": float(
                    observed_row[
                        "entropy"
                    ]
                ),
                "scenario_entropy": float(
                    scenario_row[
                        "entropy"
                    ]
                ),
                "entropy_change": float(
                    scenario_row[
                        "entropy"
                    ]
                    - observed_row[
                        "entropy"
                    ]
                ),
                "observed_effective_communities": float(
                    observed_row[
                        "effective_communities"
                    ]
                ),
                "scenario_effective_communities": float(
                    scenario_row[
                        "effective_communities"
                    ]
                ),
                "effective_communities_change": float(
                    scenario_row[
                        "effective_communities"
                    ]
                    - observed_row[
                        "effective_communities"
                    ]
                ),
                "observed_dominant_community": int(
                    observed_row[
                        "dominant_community"
                    ]
                ),
                "scenario_dominant_community": int(
                    scenario_row[
                        "dominant_community"
                    ]
                ),
                "dominant_community_changed": bool(
                    scenario_row[
                        "dominant_community"
                    ]
                    != observed_row[
                        "dominant_community"
                    ]
                ),
                "observed_dominant_share": float(
                    observed_row[
                        "dominant_community_share"
                    ]
                ),
                "scenario_dominant_share": float(
                    scenario_row[
                        "dominant_community_share"
                    ]
                ),
                "dominant_share_change": float(
                    scenario_row[
                        "dominant_community_share"
                    ]
                    - observed_row[
                        "dominant_community_share"
                    ]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ======================================================================
# PAIRWISE CHANGES
# ======================================================================

def build_pair_changes(
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate leave-one-artist-out changes in pairwise distances.
    """

    observed = (
        pairs.loc[
            pairs[
                "is_observed"
            ]
        ]
        .set_index(
            "company_pair"
        )
    )

    scenarios = pairs.loc[
        ~pairs[
            "is_observed"
        ]
    ]

    rows = []

    for _, scenario_row in scenarios.iterrows():
        company_pair = str(
            scenario_row[
                "company_pair"
            ]
        )

        observed_row = observed.loc[
            company_pair
        ]

        observed_distance = float(
            observed_row[
                "cosine_distance"
            ]
        )

        scenario_distance = float(
            scenario_row[
                "cosine_distance"
            ]
        )

        absolute_change = float(
            scenario_distance
            - observed_distance
        )

        relative_change = (
            float(
                absolute_change
                / observed_distance
            )
            if abs(
                observed_distance
            ) > EPSILON
            else np.nan
        )

        rows.append(
            {
                "scenario_id": int(
                    scenario_row[
                        "scenario_id"
                    ]
                ),
                "removed_artist": str(
                    scenario_row[
                        "removed_artist"
                    ]
                ),
                "company_a": str(
                    scenario_row[
                        "company_a"
                    ]
                ),
                "company_b": str(
                    scenario_row[
                        "company_b"
                    ]
                ),
                "company_pair": company_pair,
                "observed_distance": observed_distance,
                "scenario_distance": scenario_distance,
                "distance_change": absolute_change,
                "absolute_distance_change": abs(
                    absolute_change
                ),
                "relative_distance_change": (
                    relative_change
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ======================================================================
# RANKING CHANGES
# ======================================================================

def build_ranking_changes(
    rankings: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate company rank and distinctiveness changes.
    """

    observed = (
        rankings.loc[
            rankings[
                "is_observed"
            ]
        ]
        .set_index(
            "company"
        )
    )

    scenarios = rankings.loc[
        ~rankings[
            "is_observed"
        ]
    ]

    rows = []

    for _, scenario_row in scenarios.iterrows():
        company = str(
            scenario_row[
                "company"
            ]
        )

        observed_row = observed.loc[
            company
        ]

        observed_rank = int(
            observed_row[
                "distinctiveness_rank"
            ]
        )

        scenario_rank = int(
            scenario_row[
                "distinctiveness_rank"
            ]
        )

        observed_distance = float(
            observed_row[
                "mean_pairwise_distance"
            ]
        )

        scenario_distance = float(
            scenario_row[
                "mean_pairwise_distance"
            ]
        )

        rows.append(
            {
                "scenario_id": int(
                    scenario_row[
                        "scenario_id"
                    ]
                ),
                "removed_artist": str(
                    scenario_row[
                        "removed_artist"
                    ]
                ),
                "company": company,
                "observed_rank": observed_rank,
                "scenario_rank": scenario_rank,
                "rank_change": int(
                    scenario_rank
                    - observed_rank
                ),
                "absolute_rank_change": int(
                    abs(
                        scenario_rank
                        - observed_rank
                    )
                ),
                "rank_preserved": bool(
                    scenario_rank
                    == observed_rank
                ),
                "observed_mean_pairwise_distance": (
                    observed_distance
                ),
                "scenario_mean_pairwise_distance": (
                    scenario_distance
                ),
                "distinctiveness_change": float(
                    scenario_distance
                    - observed_distance
                ),
                "absolute_distinctiveness_change": float(
                    abs(
                        scenario_distance
                        - observed_distance
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ======================================================================
# GLOBAL SCENARIO STABILITY
# ======================================================================

def build_ranking_stability(
    rankings: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate global ordering and grouping preservation per artist removal.
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

    scenarios = rankings.loc[
        ~rankings[
            "is_observed"
        ]
    ]

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

    rows = []

    for scenario_id, scenario in scenarios.groupby(
        "scenario_id",
        sort=True,
    ):
        scenario = (
            scenario
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

        removed_artists = scenario[
            "removed_artist"
        ].unique()

        if len(
            removed_artists
        ) != 1:
            raise ValueError(
                f"Scenario {scenario_id} contains multiple artists."
            )

        scenario_order = tuple(
            scenario[
                "company"
            ].tolist()
        )

        scenario_top_two = frozenset(
            scenario.head(
                2
            )[
                "company"
            ].tolist()
        )

        scenario_bottom_two = frozenset(
            scenario.tail(
                2
            )[
                "company"
            ].tolist()
        )
        observed_rank_lookup = (
            observed.set_index(
                "company"
            )[
                "distinctiveness_rank"
            ]
        )

        scenario_rank_lookup = (
            scenario.set_index(
                "company"
            )[
                "distinctiveness_rank"
            ]
        )

        rank_displacements = (
            scenario_rank_lookup
            - observed_rank_lookup
        ).abs()

        maximum_rank_displacement = int(
            rank_displacements.max()
        )

        total_rank_displacement = int(
            rank_displacements.sum()
        )

        changed_company_count = int(
            (
                rank_displacements
                > 0
            ).sum()
        )
        rows.append(
            {
                "scenario_id": int(
                    scenario_id
                ),
                "removed_artist": str(
                    removed_artists[
                        0
                    ]
                ),
                "observed_order": "|".join(
                    observed_order
                ),
                "scenario_order": "|".join(
                    scenario_order
                ),
                "exact_order_preserved": bool(
                    scenario_order
                    == observed_order
                ),
                "top_two_set_preserved": bool(
                    scenario_top_two
                    == observed_top_two
                ),
                "bottom_two_set_preserved": bool(
                    scenario_bottom_two
                    == observed_bottom_two
                ),
                "joint_grouping_preserved": bool(
                    (
                        scenario_top_two
                        == observed_top_two
                    )
                    and
                    (
                        scenario_bottom_two
                        == observed_bottom_two
                    )
                ),
                "maximum_rank_displacement": (
                    maximum_rank_displacement
                ),
                "total_rank_displacement": (
                    total_rank_displacement
                ),
                "changed_company_count": (
                    changed_company_count
                ),
            }
        )
        
    return pd.DataFrame(
        rows
    )


# ======================================================================
# ARTIST INFLUENCE
# ======================================================================

def build_artist_influence(
    master: pd.DataFrame,
    artist_column: str,
    artist_mapping: pd.DataFrame,
    profile_changes: pd.DataFrame,
    pair_changes: pd.DataFrame,
    ranking_changes: pd.DataFrame,
    ranking_stability: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one influence summary row per removed artist.
    """

    artist_counts = (
        master.groupby(
            artist_column
        )
        .size()
        .rename(
            "artist_song_count"
        )
        .reset_index()
        .rename(
            columns={
                artist_column: "removed_artist",
            }
        )
    )

    total_songs = len(
        master
    )

    artist_counts[
        "artist_total_song_share"
    ] = (
        artist_counts[
            "artist_song_count"
        ]
        / total_songs
    )

    affected_profiles = profile_changes.loc[
        profile_changes[
            "affected_company"
        ]
    ]

    profile_summary = (
        affected_profiles.groupby(
            "removed_artist",
            as_index=False,
        )
        .agg(
            affected_company_count=(
                "company",
                "nunique",
            ),
            maximum_profile_cosine_distance=(
                "profile_cosine_distance",
                "max",
            ),
            mean_profile_cosine_distance=(
                "profile_cosine_distance",
                "mean",
            ),
            maximum_profile_l1_shift=(
                "profile_l1_shift",
                "max",
            ),
            maximum_profile_l2_shift=(
                "profile_l2_shift",
                "max",
            ),
            maximum_community_share_shift=(
                "maximum_community_share_shift",
                "max",
            ),
            maximum_removed_company_song_share=(
                "removed_company_song_share",
                "max",
            ),
            dominant_community_changes=(
                "dominant_community_changed",
                "sum",
            ),
            maximum_absolute_entropy_change=(
                "entropy_change",
                lambda values: float(
                    np.max(
                        np.abs(
                            values
                        )
                    )
                ),
            ),
        )
    )

    pair_summary = (
        pair_changes.groupby(
            "removed_artist",
            as_index=False,
        )
        .agg(
            maximum_pair_distance_change=(
                "absolute_distance_change",
                "max",
            ),
            mean_pair_distance_change=(
                "absolute_distance_change",
                "mean",
            ),
            total_pair_distance_change=(
                "absolute_distance_change",
                "sum",
            ),
        )
    )

    ranking_summary = (
        ranking_changes.groupby(
            "removed_artist",
            as_index=False,
        )
        .agg(
            changed_company_ranks=(
                "rank_preserved",
                lambda values: int(
                    (
                        ~values.astype(bool)
                    ).sum()
                ),
            ),
            maximum_rank_change=(
                "absolute_rank_change",
                "max",
            ),
            maximum_distinctiveness_change=(
                "absolute_distinctiveness_change",
                "max",
            ),
            mean_distinctiveness_change=(
                "absolute_distinctiveness_change",
                "mean",
            ),
        )
    )

    mapping = artist_mapping.rename(
        columns={
            artist_mapping.columns[
                0
            ]: "removed_artist",
        }
    )

    influence = (
        artist_counts
        .merge(
            mapping,
            on="removed_artist",
            how="left",
            validate="one_to_one",
        )
        .merge(
            profile_summary,
            on="removed_artist",
            how="left",
            validate="one_to_one",
        )
        .merge(
            pair_summary,
            on="removed_artist",
            how="left",
            validate="one_to_one",
        )
        .merge(
            ranking_summary,
            on="removed_artist",
            how="left",
            validate="one_to_one",
        )
        .merge(
            ranking_stability[
                [
                    "removed_artist",
                    "exact_order_preserved",
                    "top_two_set_preserved",
                    "bottom_two_set_preserved",
                    "joint_grouping_preserved",
                    "scenario_order",
                ]
            ],
            on="removed_artist",
            how="left",
            validate="one_to_one",
        )
    )

    numeric_fill_columns = [
        "affected_company_count",
        "maximum_profile_cosine_distance",
        "mean_profile_cosine_distance",
        "maximum_profile_l1_shift",
        "maximum_profile_l2_shift",
        "maximum_community_share_shift",
        "maximum_removed_company_song_share",
        "dominant_community_changes",
        "maximum_absolute_entropy_change",
        "maximum_pair_distance_change",
        "mean_pair_distance_change",
        "total_pair_distance_change",
        "changed_company_ranks",
        "maximum_rank_change",
        "maximum_distinctiveness_change",
        "mean_distinctiveness_change",
    ]

    for column in numeric_fill_columns:
        influence[
            column
        ] = influence[
            column
        ].fillna(
            0
        )

    influence[
        "artist_influence_score"
    ] = (
        influence[
            "maximum_profile_cosine_distance"
        ]
        + influence[
            "maximum_pair_distance_change"
        ]
        + influence[
            "maximum_distinctiveness_change"
        ]
        + (
            0.05
            * influence[
                "maximum_rank_change"
            ]
        )
    )

    return (
        influence
        .sort_values(
            [
                "artist_influence_score",
                "artist_song_count",
                "removed_artist",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )


# ======================================================================
# COMPANY AND PAIR SUMMARIES
# ======================================================================

def classify_profile_robustness(
    maximum_shift: float,
    rank_preservation_probability: float,
) -> str:
    """
    Classify company robustness to individual-artist removal.
    """

    if (
        maximum_shift <= PROFILE_SHIFT_SMALL
        and rank_preservation_probability >= 0.90
    ):
        return "highly_robust"

    if (
        maximum_shift <= PROFILE_SHIFT_MODERATE
        and rank_preservation_probability >= 0.75
    ):
        return "robust"

    if rank_preservation_probability >= 0.50:
        return "moderately_robust"

    return "artist_sensitive"


def classify_pair_robustness(
    maximum_change: float,
) -> str:
    """
    Classify pairwise-distance robustness.
    """

    if maximum_change <= PAIR_SHIFT_SMALL:
        return "highly_robust"

    if maximum_change <= PAIR_SHIFT_MODERATE:
        return "robust"

    if maximum_change <= 0.050:
        return "moderately_robust"

    return "artist_sensitive"


def build_company_summary(
    profile_changes: pd.DataFrame,
    ranking_changes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarise profile and ranking sensitivity for each company.
    """

    affected_profiles = profile_changes.loc[
        profile_changes[
            "affected_company"
        ]
    ]

    profile_summary = (
        affected_profiles.groupby(
            "company",
            as_index=False,
        )
        .agg(
            n_artist_removals=(
                "removed_artist",
                "nunique",
            ),
            maximum_profile_cosine_distance=(
                "profile_cosine_distance",
                "max",
            ),
            mean_profile_cosine_distance=(
                "profile_cosine_distance",
                "mean",
            ),
            median_profile_cosine_distance=(
                "profile_cosine_distance",
                "median",
            ),
            maximum_profile_l1_shift=(
                "profile_l1_shift",
                "max",
            ),
            maximum_community_share_shift=(
                "maximum_community_share_shift",
                "max",
            ),
            dominant_community_change_probability=(
                "dominant_community_changed",
                "mean",
            ),
            maximum_removed_company_song_share=(
                "removed_company_song_share",
                "max",
            ),
        )
    )

    ranking_summary = (
        ranking_changes.groupby(
            "company",
            as_index=False,
        )
        .agg(
            observed_rank=(
                "observed_rank",
                "first",
            ),
            rank_preservation_probability=(
                "rank_preserved",
                "mean",
            ),
            mean_absolute_rank_change=(
                "absolute_rank_change",
                "mean",
            ),
            maximum_rank_change=(
                "absolute_rank_change",
                "max",
            ),
            maximum_distinctiveness_change=(
                "absolute_distinctiveness_change",
                "max",
            ),
            mean_distinctiveness_change=(
                "absolute_distinctiveness_change",
                "mean",
            ),
        )
    )

    summary = profile_summary.merge(
        ranking_summary,
        on="company",
        how="outer",
        validate="one_to_one",
    )

    summary[
        "robustness_conclusion"
    ] = summary.apply(
        lambda row: classify_profile_robustness(
            maximum_shift=float(
                row[
                    "maximum_profile_cosine_distance"
                ]
            ),
            rank_preservation_probability=float(
                row[
                    "rank_preservation_probability"
                ]
            ),
        ),
        axis=1,
    )

    return (
        summary
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


def build_pair_summary(
    pair_changes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarise sensitivity of each pairwise distance.
    """

    summary = (
        pair_changes.groupby(
            "company_pair",
            as_index=False,
        )
        .agg(
            observed_distance=(
                "observed_distance",
                "first",
            ),
            mean_scenario_distance=(
                "scenario_distance",
                "mean",
            ),
            minimum_scenario_distance=(
                "scenario_distance",
                "min",
            ),
            maximum_scenario_distance=(
                "scenario_distance",
                "max",
            ),
            mean_absolute_distance_change=(
                "absolute_distance_change",
                "mean",
            ),
            median_absolute_distance_change=(
                "absolute_distance_change",
                "median",
            ),
            maximum_absolute_distance_change=(
                "absolute_distance_change",
                "max",
            ),
            distance_change_std=(
                "distance_change",
                "std",
            ),
        )
    )

    summary[
        "robustness_conclusion"
    ] = summary[
        "maximum_absolute_distance_change"
    ].apply(
        classify_pair_robustness
    )

    return (
        summary
        .sort_values(
            [
                "observed_distance",
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
# PUBLICATION SUMMARY
# ======================================================================

def classify_global_order_stability(
    probability: float,
) -> str:
    """
    Classify exact-order preservation across artist removals.
    """

    if probability >= 0.95:
        return "highly_robust"

    if probability >= 0.80:
        return "robust"

    if probability >= 0.60:
        return "moderately_robust"

    return "artist_sensitive"


def build_publication_summary(
    company_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    pair_changes: pd.DataFrame,
    ranking_stability: pd.DataFrame,
    artist_influence: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct the compact reviewer-facing LOAO summary.
    """

    rows = []

    exact_order_probability = float(
        ranking_stability[
            "exact_order_preserved"
        ].mean()
    )

    top_two_probability = float(
        ranking_stability[
            "top_two_set_preserved"
        ].mean()
    )

    bottom_two_probability = float(
        ranking_stability[
            "bottom_two_set_preserved"
        ].mean()
    )

    joint_group_probability = float(
        ranking_stability[
            "joint_grouping_preserved"
        ].mean()
    )

    maximum_rank_displacement = int(
        ranking_stability[
            "maximum_rank_displacement"
        ].max()
    )

    rows.append(
        {
            "analysis": (
                "Global leave-one-artist-out ranking"
            ),
            "entity": "All companies",
            "finding_tested": (
                "Observed complete company ordering"
            ),
            "observed_value": np.nan,
            "primary_robustness_probability": (
                exact_order_probability
            ),
            "secondary_test": (
                "Observed top-two and bottom-two grouping"
            ),
            "secondary_probability": (
                joint_group_probability
            ),
            "maximum_change": float(
                maximum_rank_displacement
            ),
            "mean_change": np.nan,
            "most_influential_artist": (
                artist_influence.iloc[
                    0
                ][
                    "removed_artist"
                ]
            ),
            "conclusion": (
                classify_global_order_stability(
                    exact_order_probability
                )
            ),
        }
    )

    for _, company_row in company_summary.iterrows():
        company = str(
            company_row[
                "company"
            ]
        )

        company_artists = (
            artist_influence.loc[
                artist_influence[
                    "companies"
                ]
                .astype(str)
                .str.split(
                    "|",
                    regex=False,
                )
                .apply(
                    lambda values: (
                        company
                        in values
                    )
                )
            ]
        )

        if company_artists.empty:
            most_influential_artist = ""
        else:
            most_influential_artist = str(
                company_artists.iloc[
                    0
                ][
                    "removed_artist"
                ]
            )

        rows.append(
            {
                "analysis": (
                    "Company profile robustness"
                ),
                "entity": company,
                "finding_tested": (
                    "Company semantic fingerprint and rank"
                ),
                "observed_value": float(
                    company_row[
                        "observed_rank"
                    ]
                ),
                "primary_robustness_probability": float(
                    company_row[
                        "rank_preservation_probability"
                    ]
                ),
                "secondary_test": (
                    "Dominant community preserved"
                ),
                "secondary_probability": float(
                    1.0
                    - company_row[
                        "dominant_community_change_probability"
                    ]
                ),
                "maximum_change": float(
                    company_row[
                        "maximum_profile_cosine_distance"
                    ]
                ),
                "mean_change": float(
                    company_row[
                        "mean_profile_cosine_distance"
                    ]
                ),
                "most_influential_artist": (
                    most_influential_artist
                ),
                "conclusion": str(
                    company_row[
                        "robustness_conclusion"
                    ]
                ),
            }
        )

    for _, pair_row in pair_summary.iterrows():
        company_pair = str(
            pair_row[
                "company_pair"
            ]
        )

        pair_specific_changes = (
            pair_changes.loc[
                pair_changes[
                    "company_pair"
                ] == company_pair
            ]
            .sort_values(
                [
                    "absolute_distance_change",
                    "removed_artist",
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

        if pair_specific_changes.empty:
            most_influential_artist = ""
        else:
            most_influential_artist = str(
                pair_specific_changes.iloc[
                    0
                ][
                    "removed_artist"
                ]
            )

        

        rows.append(
            {
                "analysis": (
                    "Pairwise company distance robustness"
                ),
                "entity": company_pair,
                "finding_tested": (
                    "Observed cosine distance"
                ),
                "observed_value": float(
                    pair_row[
                        "observed_distance"
                    ]
                ),
                "primary_robustness_probability": np.nan,
                "secondary_test": (
                    "Maximum absolute distance change"
                ),
                "secondary_probability": np.nan,
                "maximum_change": float(
                    pair_row[
                        "maximum_absolute_distance_change"
                    ]
                ),
                "mean_change": float(
                    pair_row[
                        "mean_absolute_distance_change"
                    ]
                ),
                "most_influential_artist": (
                    most_influential_artist
                ),
                "conclusion": str(
                    pair_row[
                        "robustness_conclusion"
                    ]
                ),
            }
        )

    global_row = rows[
        0
    ]

    global_row[
        "top_two_set_probability"
    ] = top_two_probability

    global_row[
        "bottom_two_set_probability"
    ] = bottom_two_probability

    return pd.DataFrame(
        rows
    )


# ======================================================================
# CONSOLE OUTPUT
# ======================================================================

def print_results(
    master: pd.DataFrame,
    artist_column: str,
    companies: list[str],
    communities: list[int],
    company_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    ranking_stability: pd.DataFrame,
    artist_influence: pd.DataFrame,
    publication_summary: pd.DataFrame,
) -> None:
    """
    Print the principal leave-one-artist-out results.
    """

    print_heading(
        "WP6 : LEAVE-ONE-ARTIST-OUT ROBUSTNESS"
    )

    print()
    print(
        f"Master dataset        : {MASTER_PATH}"
    )
    print(
        f"Eligible master rows  : {len(master)}"
    )
    print(
        f"Artist column         : {artist_column}"
    )
    print(
        "Artists analysed      : "
        f"{master[artist_column].nunique()}"
    )
    print(
        f"Companies analysed    : {len(companies)}"
    )
    print(
        f"Communities analysed  : {len(communities)}"
    )

    print_heading(
        "GLOBAL RANKING STABILITY"
    )

    observed_order = (
        ranking_stability[
            "observed_order"
        ].iloc[
            0
        ]
    )

    exact_order_probability = float(
        ranking_stability[
            "exact_order_preserved"
        ].mean()
    )

    top_two_probability = float(
        ranking_stability[
            "top_two_set_preserved"
        ].mean()
    )

    bottom_two_probability = float(
        ranking_stability[
            "bottom_two_set_preserved"
        ].mean()
    )

    joint_group_probability = float(
        ranking_stability[
            "joint_grouping_preserved"
        ].mean()
    )

    print(
        f"Observed order           : {observed_order}"
    )
    print(
        "Exact-order preservation : "
        f"{exact_order_probability:.4f}"
    )
    print(
        "Top-two-set preservation : "
        f"{top_two_probability:.4f}"
    )
    print(
        "Bottom-two preservation  : "
        f"{bottom_two_probability:.4f}"
    )
    print(
        "Joint grouping preserved : "
        f"{joint_group_probability:.4f}"
    )

    print_heading(
        "COMPANY ROBUSTNESS SUMMARY"
    )

    company_columns = [
        "company",
        "observed_rank",
        "n_artist_removals",
        "rank_preservation_probability",
        "maximum_rank_change",
        "maximum_profile_cosine_distance",
        "mean_profile_cosine_distance",
        "dominant_community_change_probability",
        "maximum_removed_company_song_share",
        "robustness_conclusion",
    ]

    print(
        company_summary[
            company_columns
        ].to_string(
            index=False
        )
    )

    print_heading(
        "PAIRWISE DISTANCE ROBUSTNESS"
    )

    pair_columns = [
        "company_pair",
        "observed_distance",
        "mean_absolute_distance_change",
        "maximum_absolute_distance_change",
        "minimum_scenario_distance",
        "maximum_scenario_distance",
        "robustness_conclusion",
    ]

    print(
        pair_summary[
            pair_columns
        ].to_string(
            index=False
        )
    )

    print_heading(
        "MOST INFLUENTIAL ARTIST REMOVALS"
    )

    influence_columns = [
        "removed_artist",
        "companies",
        "artist_song_count",
        "maximum_removed_company_song_share",
        "maximum_profile_cosine_distance",
        "maximum_pair_distance_change",
        "maximum_rank_change",
        "exact_order_preserved",
        "scenario_order",
        "artist_influence_score",
    ]

    print(
        artist_influence[
            influence_columns
        ]
        .head(
            15
        )
        .to_string(
            index=False
        )
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
    Execute WP6c leave-one-artist-out robustness analysis.
    """

    print_heading(
        "WP6 : LEAVE-ONE-ARTIST-OUT ROBUSTNESS"
    )

    if not MASTER_PATH.exists():
        raise FileNotFoundError(
            "The authoritative master dataset does not exist: "
            f"{MASTER_PATH}"
        )

    master_raw = pd.read_parquet(
        MASTER_PATH
    )

    artist_column = detect_artist_column(
        master_raw
    )

    validate_master_dataset(
        dataframe=master_raw,
        artist_column=artist_column,
    )

    master = prepare_master_dataset(
        dataframe=master_raw,
        artist_column=artist_column,
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

    artists = sorted(
        master[
            artist_column
        ]
        .unique()
        .tolist()
    )

    if len(
        companies
    ) < 2:
        raise ValueError(
            "At least two companies are required."
        )

    if len(
        artists
    ) < 2:
        raise ValueError(
            "At least two artists are required."
        )

    artist_mapping = (
        validate_artist_company_mapping(
            dataframe=master,
            artist_column=artist_column,
        )
    )

    multi_company_artists = (
        artist_mapping.loc[
            artist_mapping[
                "n_companies"
            ] > 1
        ]
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
        f"Artist column         : {artist_column}"
    )
    print(
        f"Artists analysed      : {len(artists)}"
    )
    print(
        f"Companies analysed    : {len(companies)}"
    )
    print(
        f"Communities analysed  : {len(communities)}"
    )
    print(
        "Multi-company artists : "
        f"{len(multi_company_artists)}"
    )

    (
        profiles,
        pairs,
        rankings,
    ) = run_leave_one_artist_out(
        master=master,
        artist_column=artist_column,
        companies=companies,
        communities=communities,
    )

    profile_changes = (
        build_profile_changes(
            profiles=profiles,
            master=master,
            artist_column=artist_column,
            communities=communities,
        )
    )

    pair_changes = (
        build_pair_changes(
            pairs=pairs
        )
    )

    ranking_changes = (
        build_ranking_changes(
            rankings=rankings
        )
    )

    ranking_stability = (
        build_ranking_stability(
            rankings=rankings
        )
    )

    artist_influence = (
        build_artist_influence(
            master=master,
            artist_column=artist_column,
            artist_mapping=artist_mapping,
            profile_changes=profile_changes,
            pair_changes=pair_changes,
            ranking_changes=ranking_changes,
            ranking_stability=ranking_stability,
        )
    )

    company_summary = (
        build_company_summary(
            profile_changes=profile_changes,
            ranking_changes=ranking_changes,
        )
    )

    pair_summary = (
        build_pair_summary(
            pair_changes=pair_changes
        )
    )

    publication_summary = (
        build_publication_summary(
            company_summary=company_summary,
            pair_summary=pair_summary,
            pair_changes=pair_changes,
            ranking_stability=ranking_stability,
            artist_influence=artist_influence,
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    profiles.to_csv(
        LOAO_PROFILES_PATH,
        index=False,
    )

    pairs.to_csv(
        LOAO_PAIRS_PATH,
        index=False,
    )

    rankings.to_csv(
        LOAO_RANKINGS_PATH,
        index=False,
    )

    profile_changes.to_csv(
        PROFILE_CHANGES_PATH,
        index=False,
    )

    pair_changes.to_csv(
        PAIR_CHANGES_PATH,
        index=False,
    )

    ranking_changes.to_csv(
        RANKING_CHANGES_PATH,
        index=False,
    )

    artist_influence.to_csv(
        ARTIST_INFLUENCE_PATH,
        index=False,
    )

    company_summary.to_csv(
        COMPANY_SUMMARY_PATH,
        index=False,
    )

    pair_summary.to_csv(
        PAIR_SUMMARY_PATH,
        index=False,
    )

    ranking_stability.to_csv(
        RANKING_STABILITY_PATH,
        index=False,
    )

    publication_summary.to_csv(
        PUBLICATION_SUMMARY_PATH,
        index=False,
    )

    print_results(
        master=master,
        artist_column=artist_column,
        companies=companies,
        communities=communities,
        company_summary=company_summary,
        pair_summary=pair_summary,
        ranking_stability=ranking_stability,
        artist_influence=artist_influence,
        publication_summary=publication_summary,
    )

    print_heading(
        "FILES WRITTEN"
    )

    for output_path in [
        LOAO_PROFILES_PATH,
        LOAO_PAIRS_PATH,
        LOAO_RANKINGS_PATH,
        PROFILE_CHANGES_PATH,
        PAIR_CHANGES_PATH,
        RANKING_CHANGES_PATH,
        ARTIST_INFLUENCE_PATH,
        COMPANY_SUMMARY_PATH,
        PAIR_SUMMARY_PATH,
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