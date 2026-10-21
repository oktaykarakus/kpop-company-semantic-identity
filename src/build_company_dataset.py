from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "company_semantics"
    / "notebooks"
    / "archive_student"
    / "datapreprocessing.ipynb"
)

LYRICS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed_revision1"
    / "lyrics.parquet"
)

COMMUNITIES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed_revision1"
    / "lyric_communities_revision1.parquet"
)

METRICS_PATH = (
    PROJECT_ROOT
    / "revision"
    / "WP4_revision1_boundary_validation"
    / "outputs"
    / "raw"
    / "boundary_metrics_all_songs.parquet"
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


# =============================================================================
# Notebook dictionary extraction
# =============================================================================

def load_notebook_source(path: Path) -> str:
    """Return all Python code cells from a Jupyter notebook as one string."""
    if not path.exists():
        raise FileNotFoundError(
            f"Student notebook not found:\n{path}\n\n"
            "Copy datapreprocessing.ipynb into "
            "company_semantics/notebooks/archive_student/."
        )

    with path.open("r", encoding="utf-8") as file:
        notebook = json.load(file)

    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ]

    return "\n\n".join(code_cells)


def extract_literal_assignment(source: str, variable_name: str) -> Any:
    """
    Safely extract a literal dictionary/list assignment from Python source.

    No notebook code is executed.
    """
    syntax_tree = ast.parse(source)

    for node in ast.walk(syntax_tree):
        if not isinstance(node, ast.Assign):
            continue

        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == variable_name:
                try:
                    return ast.literal_eval(node.value)
                except (ValueError, TypeError, SyntaxError) as exc:
                    raise ValueError(
                        f"{variable_name} was found but could not be read "
                        "as a literal value."
                    ) from exc

    raise KeyError(f"Could not find {variable_name} in {NOTEBOOK_PATH.name}.")


# =============================================================================
# Data loading and validation
# =============================================================================

def require_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    dataframe_name: str,
) -> None:
    missing = required_columns.difference(dataframe.columns)

    if missing:
        raise ValueError(
            f"{dataframe_name} is missing required columns: "
            f"{sorted(missing)}"
        )


def load_epj_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the final EPJ corpus, partition, and per-song metrics."""
    for path in (LYRICS_PATH, COMMUNITIES_PATH, METRICS_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required input file not found: {path}")

    lyrics = pd.read_parquet(LYRICS_PATH)
    communities = pd.read_parquet(COMMUNITIES_PATH)
    metrics = pd.read_parquet(METRICS_PATH)

    require_columns(
        lyrics,
        {"song_uid", "title", "artist", "date"},
        "lyrics.parquet",
    )
    require_columns(
        communities,
        {"song_uid", "community"},
        "lyric_communities_revision1.parquet",
    )
    require_columns(
        metrics,
        {
            "song_uid",
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
        },
        "boundary_metrics_all_songs.parquet",
    )

    for name, dataframe in {
        "lyrics": lyrics,
        "communities": communities,
        "metrics": metrics,
    }.items():
        if dataframe["song_uid"].duplicated().any():
            duplicates = int(dataframe["song_uid"].duplicated().sum())
            raise ValueError(
                f"{name} contains {duplicates} duplicated song_uid values."
            )

    return lyrics, communities, metrics


# =============================================================================
# Mapping and review rules
# =============================================================================

def create_mapping_tables(
    name_normalise: dict[str, str],
    company_lookup: dict[str, str],
) -> None:
    """Save transparent versions of the inherited student mappings."""
    alias_table = (
        pd.DataFrame(
            name_normalise.items(),
            columns=["artist_original", "artist_normalised"],
        )
        .sort_values(["artist_normalised", "artist_original"])
        .reset_index(drop=True)
    )

    company_table = (
        pd.DataFrame(
            company_lookup.items(),
            columns=["artist_normalised", "company"],
        )
        .sort_values(["company", "artist_normalised"])
        .reset_index(drop=True)
    )

    alias_table.to_csv(
        OUTPUT_DIR / "artist_alias_mapping_student.csv",
        index=False,
    )

    company_table.to_csv(
        OUTPUT_DIR / "big4_company_lookup_student.csv",
        index=False,
    )


def add_known_collision_flags(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Flag historically ambiguous artist-name matches.

    These rows are not silently deleted. They are excluded from the default
    analysis dataset until their identity is manually confirmed.
    """
    result = dataframe.copy()

    result["mapping_review_required"] = False
    result["mapping_review_reason"] = pd.NA

    review_rules = [
        (
            result["artist_normalised"].eq("LISA")
            & result["release_year"].lt(2016),
            "LISA before 2016 may refer to a different Korean artist.",
        ),
        (
            result["artist_normalised"].eq("SUHO")
            & result["release_year"].lt(2020),
            "SUHO before 2020 may not be EXO member Kim Jun-myeon.",
        ),
        (
            result["artist_normalised"].eq("Kang Seungyoon")
            & result["release_year"].lt(2012),
            "Kang Seungyoon before 2012 may refer to another artist.",
        ),
    ]

    for mask, reason in review_rules:
        result.loc[mask, "mapping_review_required"] = True
        result.loc[mask, "mapping_review_reason"] = reason

    return result


# =============================================================================
# Master dataset construction
# =============================================================================

def build_company_master() -> pd.DataFrame:
    notebook_source = load_notebook_source(NOTEBOOK_PATH)

    name_normalise = extract_literal_assignment(
        notebook_source,
        "NAME_NORMALISE",
    )
    company_lookup = extract_literal_assignment(
        notebook_source,
        "BIG4_LOOKUP",
    )

    if not isinstance(name_normalise, dict):
        raise TypeError("NAME_NORMALISE is not a dictionary.")

    if not isinstance(company_lookup, dict):
        raise TypeError("BIG4_LOOKUP is not a dictionary.")

    print("=" * 78)
    print("STUDENT MAPPINGS")
    print("=" * 78)
    print(f"Artist aliases: {len(name_normalise):,}")
    print(f"Big Four lookup entries: {len(company_lookup):,}")

    create_mapping_tables(name_normalise, company_lookup)

    lyrics, communities, metrics = load_epj_data()

    print("\n" + "=" * 78)
    print("FINAL EPJ INPUTS")
    print("=" * 78)
    print(f"Lyrics:      {len(lyrics):,}")
    print(f"Communities: {len(communities):,}")
    print(f"Metrics:     {len(metrics):,}")

    # The final metrics already contain the same community assignment.
    # Retain one copy and verify agreement explicitly.
    metrics = metrics.rename(
        columns={"community": "community_metrics"}
    )

    master = (
        lyrics
        .merge(
            communities,
            on="song_uid",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            metrics,
            on="song_uid",
            how="inner",
            validate="one_to_one",
        )
    )

    if len(master) != len(lyrics):
        raise RuntimeError(
            f"Master merge returned {len(master):,} rows; "
            f"expected {len(lyrics):,}."
        )

    community_mismatch = (
        master["community"]
        .ne(master["community_metrics"])
        .sum()
    )

    if community_mismatch:
        raise RuntimeError(
            f"Community mismatch in {community_mismatch:,} songs."
        )

    master = master.drop(columns="community_metrics")

    # Parse the release year conservatively from the existing date field.
    master["release_year"] = pd.to_numeric(
        master["date"].astype(str).str.extract(r"(\d{4})")[0],
        errors="coerce",
    ).astype("Int64")

    # Apply the student's alias normalisation and static company mapping.
    master["artist_normalised"] = (
        master["artist"]
        .map(name_normalise)
        .fillna(master["artist"])
    )

    master["company"] = master["artist_normalised"].map(company_lookup)

    master["mapping_status"] = master["company"].notna().map(
        {
            True: "matched_student_lookup",
            False: "not_big4_or_unmapped",
        }
    )

    master["mapping_basis"] = "static_student_lookup"
    master["company_at_release_verified"] = False

    master = add_known_collision_flags(master)

    master["analysis_eligible"] = (
        master["company"].notna()
        & ~master["mapping_review_required"]
    )

    # Add broad time bands only as convenient descriptors.
    master["release_period"] = pd.cut(
        master["release_year"],
        bins=[-float("inf"), 2004, 2009, 2014, 2019, float("inf")],
        labels=[
            "<=2004",
            "2005-2009",
            "2010-2014",
            "2015-2019",
            ">=2020",
        ],
    )

    preferred_columns = [
        # Identity
        "song_uid",
        "title",
        "artist",
        "artist_normalised",
        "date",
        "release_year",
        "release_period",
        # Institution
        "company",
        "mapping_status",
        "mapping_basis",
        "company_at_release_verified",
        "mapping_review_required",
        "mapping_review_reason",
        "analysis_eligible",
        # Semantic structure
        "community",
        "degree",
        "weighted_degree",
        "internal_strength",
        "external_strength",
        "external_strength_fraction",
        "participation_coefficient",
        "community_entropy",
        "communities_reached",
        "external_communities_reached",
        "mean_neighbour_cosine",
        # Novelty
        "local_novelty",
        "max_centroid_cosine",
        "second_centroid_cosine",
        "centroid_margin",
        "centroid_novelty",
        "bridge_affinity_score",
        # Original corpus fields
        "lyrics",
        "lines",
        "n_lines",
        "n_chars",
        "source_file",
    ]

    available_columns = [
        column
        for column in preferred_columns
        if column in master.columns
    ]

    master = master[available_columns].copy()

    return master


# =============================================================================
# Audit outputs
# =============================================================================

def save_audits(master: pd.DataFrame) -> None:
    """Save mapping diagnostics before substantive analysis begins."""
    company_rows = master[master["company"].notna()].copy()
    eligible = master[master["analysis_eligible"]].copy()

    artist_audit = (
        master
        .groupby(
            [
                "artist",
                "artist_normalised",
                "company",
                "mapping_status",
                "mapping_review_required",
            ],
            dropna=False,
        )
        .agg(
            n_songs=("song_uid", "size"),
            min_year=("release_year", "min"),
            max_year=("release_year", "max"),
            n_communities=("community", "nunique"),
        )
        .reset_index()
        .sort_values(
            ["company", "n_songs", "artist_normalised"],
            ascending=[True, False, True],
            na_position="last",
        )
    )

    artist_audit.to_csv(
        OUTPUT_DIR / "company_mapping_artist_audit.csv",
        index=False,
    )

    ambiguous = master[master["mapping_review_required"]].copy()

    ambiguous.to_csv(
        OUTPUT_DIR / "company_mapping_manual_review.csv",
        index=False,
    )

    company_summary = (
        eligible
        .groupby("company", observed=True)
        .agg(
            n_songs=("song_uid", "size"),
            n_artists=("artist_normalised", "nunique"),
            min_year=("release_year", "min"),
            max_year=("release_year", "max"),
            n_communities=("community", "nunique"),
        )
        .reset_index()
        .sort_values("company")
    )

    company_summary.to_csv(
        OUTPUT_DIR / "company_dataset_summary.csv",
        index=False,
    )

    unmapped_top = (
        master.loc[master["company"].isna(), "artist"]
        .value_counts()
        .rename_axis("artist")
        .reset_index(name="n_songs")
    )

    unmapped_top.to_csv(
        OUTPUT_DIR / "unmapped_artist_counts.csv",
        index=False,
    )

    print("\n" + "=" * 78)
    print("MAPPING RESULTS")
    print("=" * 78)
    print(f"Final corpus songs:              {len(master):,}")
    print(f"Songs mapped to Big Four:        {len(company_rows):,}")
    print(f"Songs requiring manual review:   {len(ambiguous):,}")
    print(f"Default analysis-eligible songs: {len(eligible):,}")
    print(
        "Mapped Big Four artists:       "
        f"{eligible['artist_normalised'].nunique():,}"
    )

    print("\nCompany summary:")
    print(company_summary.to_string(index=False))

    if not ambiguous.empty:
        print("\nRows flagged for manual review:")
        print(
            ambiguous[
                [
                    "title",
                    "artist",
                    "artist_normalised",
                    "release_year",
                    "company",
                    "mapping_review_reason",
                ]
            ]
            .to_string(index=False)
        )

    print("\nMost frequent currently unmapped artists:")
    print(unmapped_top.head(20).to_string(index=False))


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    master = build_company_master()

    master_output = OUTPUT_DIR / "company_master.parquet"
    eligible_output = OUTPUT_DIR / "company_analysis_dataset.parquet"

    master.to_parquet(master_output, index=False)

    analysis_dataset = master[
        master["analysis_eligible"]
    ].copy()

    analysis_dataset.to_parquet(
        eligible_output,
        index=False,
    )

    save_audits(master)

    print("\n" + "=" * 78)
    print("SAVED")
    print("=" * 78)
    print(f"Full mapped master:\n  {master_output}")
    print(f"\nDefault analysis dataset:\n  {eligible_output}")
    print(f"\nAudit tables:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()