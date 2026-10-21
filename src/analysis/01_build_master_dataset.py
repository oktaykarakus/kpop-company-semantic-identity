from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "company_semantics"
    / "outputs"
    / "tables"
    / "company_analysis_dataset.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "company_semantics"
    / "outputs"
    / "tables"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "company_semantics"
    / "outputs"
    / "reports"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MASTER_OUTPUT = OUTPUT_DIR / "company_master_analysis_v1.parquet"
SUMMARY_OUTPUT = OUTPUT_DIR / "company_master_analysis_v1_summary.csv"
ARTIST_OUTPUT = OUTPUT_DIR / "company_master_analysis_v1_artists.csv"
COMMUNITY_OUTPUT = OUTPUT_DIR / "company_master_analysis_v1_communities.csv"
DICTIONARY_OUTPUT = REPORT_DIR / "company_master_analysis_v1_data_dictionary.csv"
VALIDATION_OUTPUT = REPORT_DIR / "company_master_analysis_v1_validation.json"


# =============================================================================
# Configuration
# =============================================================================

VALID_COMPANIES = ["HYBE", "JYP", "SM", "YG"]

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
    "local_novelty",
    "second_centroid_cosine",
    "centroid_margin",
    "centroid_novelty",
    "bridge_affinity_score",
}

PREFERRED_COLUMN_ORDER = [
    # Song identity
    "song_uid",
    "title",
    "artist",
    "artist_normalised",
    "company",
    "date",
    "year",

    # Corpus characteristics
    "n_lines",
    "n_chars",

    # Semantic community
    "community",

    # Graph structure
    "degree",
    "weighted_degree",
    "internal_strength",
    "external_strength",
    "mean_neighbour_cosine",

    # Brokerage
    "participation_coefficient",
    "community_entropy",
    "communities_reached",
    "external_communities_reached",
    "external_strength_fraction",
    "bridge_affinity_score",

    # Novelty and community assignment
    "local_novelty",
    "max_centroid_cosine",
    "second_centroid_cosine",
    "centroid_margin",
    "centroid_novelty",

    # Mapping audit fields
    "mapping_status",
    "mapping_basis",
    "company_at_release_verified",
    "mapping_review_required",
    "mapping_review_reason",
    "analysis_eligible",

    # Source fields
    "source_file",
    "lyrics",
    "lines",
]


DATA_DICTIONARY = {
    "song_uid": "Unique song identifier inherited from the validated EPJ corpus.",
    "title": "Song title.",
    "artist": "Artist name as represented in the source corpus.",
    "artist_normalised": "Canonical artist name after alias normalisation.",
    "company": "Big Four company label: HYBE, JYP, SM, or YG.",
    "date": "Original release-date field.",
    "year": "Release year extracted from the original date field.",
    "n_lines": "Number of retained lyric lines.",
    "n_chars": "Number of lyric characters.",
    "community": "Validated semantic community assignment from the EPJ graph.",
    "degree": "Unweighted graph degree.",
    "weighted_degree": "Sum of semantic edge weights.",
    "internal_strength": "Total edge strength linking the song to its own community.",
    "external_strength": "Total edge strength linking the song to other communities.",
    "mean_neighbour_cosine": "Mean cosine similarity to graph neighbours.",
    "participation_coefficient": (
        "Weighted participation coefficient measuring how evenly a song's "
        "connectivity is distributed across communities."
    ),
    "community_entropy": (
        "Entropy of the song's weighted neighbourhood-community distribution."
    ),
    "communities_reached": "Number of communities represented in the song's neighbourhood.",
    "external_communities_reached": (
        "Number of communities outside the song's own community reached by its edges."
    ),
    "external_strength_fraction": (
        "Proportion of total weighted edge strength directed outside the assigned community."
    ),
    "bridge_affinity_score": "Composite semantic-brokerage score from the EPJ analysis.",
    "local_novelty": "Novelty relative to the song's immediate semantic neighbours.",
    "max_centroid_cosine": "Highest cosine similarity to any semantic-community centroid.",
    "second_centroid_cosine": "Second-highest cosine similarity to a community centroid.",
    "centroid_margin": (
        "Difference between the highest and second-highest centroid similarities."
    ),
    "centroid_novelty": "Distance-like novelty measure relative to the closest centroid.",
    "mapping_status": "Status of the inherited Big Four artist mapping.",
    "mapping_basis": "Source or basis used for assigning the company label.",
    "company_at_release_verified": (
        "Whether the company assignment has been verified for the song's release date."
    ),
    "mapping_review_required": "Whether the artist-company assignment requires manual review.",
    "mapping_review_reason": "Reason for manual mapping review.",
    "analysis_eligible": "Whether the record is eligible for the primary company analysis.",
    "source_file": "Original source-file path in the lyric archive.",
    "lyrics": "Full retained lyric text.",
    "lines": "Processed list of lyric lines.",
}


# =============================================================================
# Helpers
# =============================================================================

def require_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)

    if missing:
        raise ValueError(
            "Input dataset is missing required columns:\n"
            + "\n".join(f"  - {column}" for column in sorted(missing))
        )


def standardise_types(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["song_uid"] = result["song_uid"].astype("string")
    result["title"] = result["title"].astype("string")
    result["artist"] = result["artist"].astype("string")
    result["artist_normalised"] = result["artist_normalised"].astype("string")

    result["company"] = pd.Categorical(
        result["company"],
        categories=VALID_COMPANIES,
        ordered=True,
    )

    result["year"] = pd.to_numeric(
        result["year"],
        errors="coerce",
    ).astype("Int64")

    result["community"] = pd.to_numeric(
        result["community"],
        errors="raise",
    ).astype("int16")

    numeric_columns = [
        "degree",
        "weighted_degree",
        "internal_strength",
        "external_strength",
        "mean_neighbour_cosine",
        "participation_coefficient",
        "community_entropy",
        "communities_reached",
        "external_communities_reached",
        "external_strength_fraction",
        "bridge_affinity_score",
        "local_novelty",
        "max_centroid_cosine",
        "second_centroid_cosine",
        "centroid_margin",
        "centroid_novelty",
        "n_lines",
        "n_chars",
    ]

    for column in numeric_columns:
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    return result


def validate_dataset(df: pd.DataFrame) -> dict:
    duplicate_song_ids = int(df["song_uid"].duplicated().sum())

    company_counts = (
        df["company"]
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )

    invalid_companies = sorted(
        set(df["company"].dropna().astype(str))
        .difference(VALID_COMPANIES)
    )

    missing_values = {
        column: int(df[column].isna().sum())
        for column in REQUIRED_COLUMNS
        if column in df.columns
    }

    metric_ranges = {}

    bounded_metrics = [
        "participation_coefficient",
        "external_strength_fraction",
        "max_centroid_cosine",
        "second_centroid_cosine",
    ]

    for column in bounded_metrics:
        values = df[column].dropna()

        metric_ranges[column] = {
            "min": float(values.min()),
            "max": float(values.max()),
        }

    validation = {
        "n_rows": int(len(df)),
        "n_unique_songs": int(df["song_uid"].nunique()),
        "n_unique_artists_original": int(df["artist"].nunique()),
        "n_unique_artists_normalised": int(df["artist_normalised"].nunique()),
        "n_communities": int(df["community"].nunique()),
        "min_year": int(df["year"].min()),
        "max_year": int(df["year"].max()),
        "duplicate_song_ids": duplicate_song_ids,
        "invalid_companies": invalid_companies,
        "company_counts": {
            str(key): int(value)
            for key, value in company_counts.items()
        },
        "missing_values": missing_values,
        "metric_ranges": metric_ranges,
    }

    if duplicate_song_ids:
        raise ValueError(
            f"Found {duplicate_song_ids} duplicated song_uid values."
        )

    if invalid_companies:
        raise ValueError(
            f"Unexpected company labels found: {invalid_companies}"
        )

    if len(df) == 0:
        raise ValueError("The analysis dataset is empty.")

    if df["company"].isna().any():
        raise ValueError(
            "The analysis dataset contains missing company labels."
        )

    if df["community"].isna().any():
        raise ValueError(
            "The analysis dataset contains missing community assignments."
        )

    return validation


def create_summary_tables(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    company_summary = (
        df.groupby("company", observed=True)
        .agg(
            n_songs=("song_uid", "size"),
            n_artists=("artist_normalised", "nunique"),
            min_year=("year", "min"),
            max_year=("year", "max"),
            n_communities=("community", "nunique"),
            mean_participation=(
                "participation_coefficient",
                "mean",
            ),
            mean_neighbour_entropy=(
                "community_entropy",
                "mean",
            ),
            mean_external_strength_fraction=(
                "external_strength_fraction",
                "mean",
            ),
            mean_external_communities=(
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
        )
        .reset_index()
    )

    artist_summary = (
        df.groupby(
            ["company", "artist_normalised"],
            observed=True,
        )
        .agg(
            n_songs=("song_uid", "size"),
            min_year=("year", "min"),
            max_year=("year", "max"),
            n_communities=("community", "nunique"),
            mean_participation=(
                "participation_coefficient",
                "mean",
            ),
            mean_centroid_novelty=(
                "centroid_novelty",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            ["company", "n_songs"],
            ascending=[True, False],
        )
    )

    community_summary = (
        df.groupby(
            ["community", "company"],
            observed=True,
        )
        .agg(
            n_songs=("song_uid", "size"),
            n_artists=("artist_normalised", "nunique"),
            mean_year=("year", "mean"),
            mean_participation=(
                "participation_coefficient",
                "mean",
            ),
            mean_centroid_novelty=(
                "centroid_novelty",
                "mean",
            ),
        )
        .reset_index()
    )

    community_summary["company_share_within_community"] = (
        community_summary["n_songs"]
        / community_summary.groupby("community")["n_songs"].transform("sum")
    )

    community_summary["community_share_within_company"] = (
        community_summary["n_songs"]
        / community_summary.groupby("company", observed=True)["n_songs"].transform("sum")
    )

    return company_summary, artist_summary, community_summary


def create_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for column in df.columns:
        rows.append(
            {
                "column": column,
                "dtype": str(df[column].dtype),
                "non_missing": int(df[column].notna().sum()),
                "missing": int(df[column].isna().sum()),
                "description": DATA_DICTIONARY.get(
                    column,
                    "No description currently defined.",
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input dataset not found:\n{INPUT_PATH}"
        )

    print("=" * 80)
    print("BUILDING COMPANY MASTER ANALYSIS DATASET")
    print("=" * 80)
    print(f"Input: {INPUT_PATH}")

    df = pd.read_parquet(INPUT_PATH)
    # ------------------------------------------------------------------
    # Create release year if only the original date field exists
    # ------------------------------------------------------------------
    if "year" not in df.columns:
        if "date" in df.columns:
            df["year"] = (
                df["date"]
                .astype(str)
                .str.extract(r"(\d{4})")[0]
                .astype("Int64")
            )
        else:
            raise ValueError(
                "Neither 'year' nor 'date' exists in the input dataset."
            )

    require_columns(df)

    df = standardise_types(df)

    # Keep only eligible Big Four rows.
    if "analysis_eligible" in df.columns:
        df = df.loc[df["analysis_eligible"]].copy()

    df = df.loc[df["company"].notna()].copy()

    available_columns = [
        column
        for column in PREFERRED_COLUMN_ORDER
        if column in df.columns
    ]

    remaining_columns = [
        column
        for column in df.columns
        if column not in available_columns
    ]

    df = df[available_columns + remaining_columns].copy()

    df = df.sort_values(
        ["company", "artist_normalised", "year", "title"],
        na_position="last",
    ).reset_index(drop=True)

    validation = validate_dataset(df)

    company_summary, artist_summary, community_summary = (
        create_summary_tables(df)
    )

    data_dictionary = create_data_dictionary(df)

    df.to_parquet(
        MASTER_OUTPUT,
        index=False,
    )

    company_summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    artist_summary.to_csv(
        ARTIST_OUTPUT,
        index=False,
    )

    community_summary.to_csv(
        COMMUNITY_OUTPUT,
        index=False,
    )

    data_dictionary.to_csv(
        DICTIONARY_OUTPUT,
        index=False,
    )

    with VALIDATION_OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            validation,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)

    print(f"Songs:              {validation['n_rows']:,}")
    print(
        "Artists:            "
        f"{validation['n_unique_artists_normalised']:,}"
    )
    print(f"Communities:         {validation['n_communities']:,}")
    print(
        "Year range:          "
        f"{validation['min_year']}–{validation['max_year']}"
    )
    print(
        "Duplicated song IDs: "
        f"{validation['duplicate_song_ids']:,}"
    )

    print("\nCompany summary:")
    print(
        company_summary[
            [
                "company",
                "n_songs",
                "n_artists",
                "min_year",
                "max_year",
                "n_communities",
                "mean_participation",
                "mean_centroid_novelty",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\n" + "=" * 80)
    print("SAVED")
    print("=" * 80)
    print(f"Master dataset:\n  {MASTER_OUTPUT}")
    print(f"Company summary:\n  {SUMMARY_OUTPUT}")
    print(f"Artist summary:\n  {ARTIST_OUTPUT}")
    print(f"Community summary:\n  {COMMUNITY_OUTPUT}")
    print(f"Data dictionary:\n  {DICTIONARY_OUTPUT}")
    print(f"Validation report:\n  {VALIDATION_OUTPUT}")


if __name__ == "__main__":
    main()