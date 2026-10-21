#!/usr/bin/env python3
"""
WP8c: Semantic Transition Explanation

This version reconstructs the rolling five-year community fingerprints directly
from the frozen master analysis table using the same global rolling-window
calendar as WP5. It does not depend on an intermediate
fingerprint CSV being present in outputs/temporal/rolling5.

For each company and adjacent valid rolling window, the script calculates:

- community probability change;
- absolute probability-change rank;
- additive community-level Jensen-Shannon divergence contribution;
- each community's percentage share of transition-level JS divergence;
- total variation, cosine similarity, and Euclidean distance;
- representative-song and artist evidence from WP8b1;
- organisational contribution context from WP8b.

Inputs
------
outputs/tables/company_master_analysis_v1.parquet
outputs/explainability/community_semantic_dictionary.csv
outputs/explainability/community_contributions.csv

Outputs
-------
outputs/explainability/semantic_transition_explanations.csv
outputs/explainability/company_transition_summary.csv
outputs/explainability/semantic_transition_metadata.json
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_VERSION = "2026-08-01-v9-wp5-window-aligned"

ROOT = Path(__file__).resolve().parents[2]

MASTER_PATH = ROOT / "outputs" / "tables" / "company_master_analysis_v1.parquet"
DICTIONARY_PATH = (
    ROOT / "outputs" / "explainability" / "community_semantic_dictionary.csv"
)
CONTRIBUTIONS_PATH = (
    ROOT / "outputs" / "explainability" / "community_contributions.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "explainability"
TRANSITIONS_PATH = OUTPUT_DIR / "semantic_transition_explanations.csv"
SUMMARY_PATH = OUTPUT_DIR / "company_transition_summary.csv"
METADATA_PATH = OUTPUT_DIR / "semantic_transition_metadata.json"

WINDOW_SIZE_YEARS = 5
MIN_SONGS_PER_WINDOW = 1

COMPANY_ORDER = ["HYBE", "JYP", "SM", "YG"]

YEAR_CANDIDATES = [
    "release_year",
    "year",
    "song_year",
    "release_date",
    "date",
    "release",
]


def banner(text: str) -> None:
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")


def clean_display_text(value: object) -> str:
    """
    Remove invisible/control characters that can erase adjacent spaces in
    terminal and CSV rendering, while preserving Korean and other Unicode text.
    """
    text = "" if value is None else str(value)

    # Remove C0/C1 control characters, including backspace, while retaining
    # ordinary printable Unicode characters.
    text = "".join(
        character
        for character in text
        if not (
            ord(character) < 32
            or 127 <= ord(character) <= 159
        )
    )

    # Normalise unusual Unicode spaces to ordinary spaces and collapse runs.
    text = (
        text.replace("\u00a0", " ")
        .replace("\u2007", " ")
        .replace("\u202f", " ")
        .replace("\u200b", "")
        .replace("\ufeff", "")
    )

    return " ".join(text.split())


def normalise_community(value: object) -> str:
    text = clean_display_text(value)
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def natural_key(value: object) -> tuple:
    parts = re.split(r"(\d+)", str(value))
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in parts
    )


def identify_year_column(frame: pd.DataFrame) -> str:
    lookup = {str(column).lower(): str(column) for column in frame.columns}

    for candidate in YEAR_CANDIDATES:
        if candidate in lookup:
            return lookup[candidate]

    possible = [
        str(column)
        for column in frame.columns
        if "year" in str(column).lower() or "date" in str(column).lower()
    ]

    if possible:
        return possible[0]

    raise ValueError(
        "No release-year or release-date column could be identified in the "
        "master dataset. Available columns:\n"
        + ", ".join(map(str, frame.columns))
    )


def extract_year(series: pd.Series, source_column: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")

    plausible_numeric = numeric.where(
        numeric.between(1900, 2100)
    )

    if plausible_numeric.notna().mean() >= 0.5:
        return plausible_numeric.round().astype("Int64")

    parsed = pd.to_datetime(series, errors="coerce")
    years = parsed.dt.year.astype("Int64")

    if years.notna().sum() == 0:
        # Final fallback: extract a four-digit year from mixed strings.
        extracted = (
            series.astype(str)
            .str.extract(r"\b(19\d{2}|20\d{2}|21\d{2})\b", expand=False)
        )
        years = pd.to_numeric(extracted, errors="coerce").astype("Int64")

    if years.notna().sum() == 0:
        raise ValueError(
            f"Could not extract valid years from column '{source_column}'."
        )

    return years


def build_rolling_fingerprints(
    eligible: pd.DataFrame,
    all_communities: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build consecutive calendar-year rolling windows.

    A window labelled end year Y contains songs released from
    Y - WINDOW_SIZE_YEARS + 1 through Y, inclusive.
    """
    records: list[dict[str, object]] = []
    inventory: list[dict[str, object]] = []

    observed_companies = list(pd.unique(eligible["company"]))
    company_rank = {
        company: index
        for index, company in enumerate(COMPANY_ORDER + observed_companies)
    }

    # Match WP5 exactly: construct one common rolling-window calendar from
    # the global analysis-year range, then retain each company-window whenever
    # it contains at least MIN_SONGS_PER_WINDOW eligible songs. This allows a
    # company to contribute to a window whose start year predates its first
    # observed release, provided the five-year interval contains company songs.
    global_minimum_year = int(eligible["analysis_year"].min())
    global_maximum_year = int(eligible["analysis_year"].max())
    final_start_year = global_maximum_year - WINDOW_SIZE_YEARS + 1

    for company in sorted(
        observed_companies,
        key=lambda value: company_rank.get(value, 10**9),
    ):
        company_data = eligible.loc[
            eligible["company"].eq(company)
        ].copy()

        for start_year in range(
            global_minimum_year,
            final_start_year + 1,
        ):
            end_year = start_year + WINDOW_SIZE_YEARS - 1

            window_data = company_data.loc[
                company_data["analysis_year"].between(
                    start_year,
                    end_year,
                    inclusive="both",
                )
            ].copy()

            n_songs = int(len(window_data))
            valid_window = n_songs >= MIN_SONGS_PER_WINDOW

            inventory.append(
                {
                    "company": company,
                    "window_start_year": start_year,
                    "window_end_year": end_year,
                    "window_label": f"{start_year}-{end_year}",
                    "n_songs": n_songs,
                    "valid_window": valid_window,
                }
            )

            if not valid_window:
                continue

            counts = (
                window_data["community"]
                .value_counts()
                .reindex(all_communities, fill_value=0)
            )

            probabilities = counts / counts.sum()

            for community in all_communities:
                records.append(
                    {
                        "company": company,
                        "window_start_year": start_year,
                        "window_end_year": end_year,
                        "window_label": f"{start_year}-{end_year}",
                        "n_songs_in_window": n_songs,
                        "community": community,
                        "community_count": int(counts.loc[community]),
                        "probability": float(probabilities.loc[community]),
                    }
                )

    fingerprints = pd.DataFrame(records)
    window_inventory = pd.DataFrame(inventory)

    if fingerprints.empty:
        raise ValueError(
            "No valid rolling windows were produced. Consider checking the "
            "year field or MIN_SONGS_PER_WINDOW."
        )

    return fingerprints, window_inventory



def build_company_community_evidence(
    eligible: pd.DataFrame,
    max_songs: int = 5,
    max_artists: int = 5,
) -> pd.DataFrame:
    """
    Build company-specific representative-song evidence for each semantic
    community.

    Songs are ranked by max_centroid_cosine when available, followed by
    weighted_degree and degree as deterministic tie-breakers. If those graph
    columns are unavailable, catalogue order is used as a final fallback.
    """
    title_candidates = ["title", "song_title", "track_name", "song", "track"]
    artist_candidates = ["artist", "artist_name", "performer", "artists"]

    def first_existing(candidates: list[str]) -> str | None:
        lookup = {str(column).lower(): str(column) for column in eligible.columns}
        for candidate in candidates:
            if candidate in lookup:
                return lookup[candidate]
        return None

    title_column = first_existing(title_candidates)
    artist_column = first_existing(artist_candidates)

    if title_column is None or artist_column is None:
        raise ValueError(
            "Company-specific representative evidence requires song-title and "
            "artist columns in the master dataset."
        )

    ranking_columns = [
        column
        for column in [
            "max_centroid_cosine",
            "weighted_degree",
            "degree",
        ]
        if column in eligible.columns
    ]

    records: list[dict[str, object]] = []

    for (company, community), group in eligible.groupby(
        ["company", "community"],
        observed=True,
        sort=False,
    ):
        ranked = group.copy()

        sort_columns: list[str] = []
        ascending: list[bool] = []

        for column in ranking_columns:
            ranked[column] = pd.to_numeric(
                ranked[column],
                errors="coerce",
            )
            sort_columns.append(column)
            ascending.append(False)

        sort_columns.extend([title_column, artist_column])
        ascending.extend([True, True])

        ranked = ranked.sort_values(
            sort_columns,
            ascending=ascending,
            kind="stable",
            na_position="last",
        )

        song_evidence: list[str] = []
        seen_songs: set[tuple[str, str]] = set()

        for row in ranked.itertuples(index=False):
            title = clean_display_text(
                getattr(row, title_column)
            )
            artist = clean_display_text(
                getattr(row, artist_column)
            )

            if not title or title.lower() == "nan":
                continue
            if not artist or artist.lower() == "nan":
                continue

            key = (title.casefold(), artist.casefold())
            if key in seen_songs:
                continue

            seen_songs.add(key)
            song_evidence.append(f"{title} — {artist}")

            if len(song_evidence) >= max_songs:
                break

        artist_counts = (
            group[artist_column]
            .dropna()
            .astype(str)
            .str.strip()
        )
        artist_counts = artist_counts[
            artist_counts.ne("")
            & artist_counts.str.lower().ne("nan")
        ].value_counts()

        representative_artists = "; ".join(
            artist_counts.head(max_artists).index.tolist()
        )

        records.append(
            {
                "company": company,
                "community": normalise_community(community),
                "company_community_representative_songs": "; ".join(song_evidence),
                "company_community_representative_artists": representative_artists,
                "company_community_n_songs": int(len(group)),
                "company_community_n_artists": int(artist_counts.size),
            }
        )

    return pd.DataFrame(records)


def js_components(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    midpoint = 0.5 * (p + q)
    result = np.zeros_like(p, dtype=float)

    positive_p = p > 0
    positive_q = q > 0

    result[positive_p] += (
        0.5
        * p[positive_p]
        * np.log2(p[positive_p] / midpoint[positive_p])
    )
    result[positive_q] += (
        0.5
        * q[positive_q]
        * np.log2(q[positive_q] / midpoint[positive_q])
    )

    return result


def cosine_similarity(p: np.ndarray, q: np.ndarray) -> float:
    denominator = float(np.linalg.norm(p) * np.linalg.norm(q))
    if denominator == 0:
        return np.nan
    return float(np.dot(p, q) / denominator)


def first_representative_song(value: object) -> str:
    cleaned = clean_display_text(value)
    if not cleaned:
        return ""
    return clean_display_text(cleaned.split(";")[0])


def describe_rows(rows: pd.DataFrame) -> str:
    descriptions: list[str] = []

    for row in rows.itertuples(index=False):
        evidence = first_representative_song(
            getattr(
                row,
                "company_community_representative_songs",
                "",
            )
        )
        if not evidence:
            evidence = first_representative_song(
                getattr(row, "representative_songs", "")
            )

        community = clean_display_text(row.community)
        label = f"Community {community}"
        if evidence:
            label += f" [{evidence}]"

        descriptions.append(
            f"{label}: {row.probability_change:+.4f}"
        )

    return " | ".join(descriptions)


def normalise_narrative_spacing(text: str) -> str:
    """
    Enforce publication-safe spacing around generated community references.

    This final rendering pass is deliberately independent of the underlying
    Unicode content, so strings such as ``Community6[...`` and
    ``Community0 explained`` are normalised even if an input field contains
    unusual invisible characters.
    """
    cleaned = clean_display_text(text)

    cleaned = re.sub(
        r"\bCommunity\s*(\d+)\s*\[",
        r"Community \1 [",
        cleaned,
    )
    cleaned = re.sub(
        r"\bCommunity\s*(\d+)\s+explained\b",
        r"Community \1 explained",
        cleaned,
    )
    cleaned = re.sub(
        r"\bCommunity\s*(\d+)\s*:",
        r"Community \1:",
        cleaned,
    )
    cleaned = re.sub(
        r"\bCommunities\s+",
        "Communities ",
        cleaned,
    )

    return cleaned


def transition_explanation(group: pd.DataFrame) -> str:
    company = str(group["company"].iloc[0])
    previous = str(group["previous_window"].iloc[0])
    current = str(group["current_window"].iloc[0])

    increases = (
        group.loc[group["probability_change"] > 0]
        .nlargest(3, "probability_change")
    )
    decreases = (
        group.loc[group["probability_change"] < 0]
        .nsmallest(3, "probability_change")
    )

    parts = [f"{company} shifted from {previous} to {current}."]

    if not increases.empty:
        parts.append(
            "Largest semantic increases: "
            + describe_rows(increases)
            + "."
        )

    if not decreases.empty:
        parts.append(
            "Largest semantic decreases: "
            + describe_rows(decreases)
            + "."
        )

    maximum_component = float(group["js_component"].max())
    tolerance = max(1e-12, abs(maximum_component) * 1e-9)

    dominant_rows = group.loc[
        np.isclose(
            group["js_component"].to_numpy(dtype=float),
            maximum_component,
            rtol=1e-9,
            atol=tolerance,
        )
    ].sort_values("community", key=lambda series: series.map(natural_key))

    dominant_communities = [
        clean_display_text(value)
        for value in dominant_rows["community"].tolist()
    ]
    dominant_share = float(dominant_rows["js_share_pct"].iloc[0])

    if len(dominant_communities) == 1:
        parts.append(
            f"Community {dominant_communities[0]} explained the largest share "
            f"of distributional change "
            f"({dominant_share:.1f}% of JS divergence)."
        )
    else:
        if len(dominant_communities) == 2:
            community_text = " and ".join(dominant_communities)
        else:
            community_text = (
                ", ".join(dominant_communities[:-1])
                + f", and {dominant_communities[-1]}"
            )

        parts.append(
            f"Communities {community_text} jointly made the largest individual "
            f"contributions, each explaining {dominant_share:.1f}% of "
            f"transition-level JS divergence."
        )

    return normalise_narrative_spacing(" ".join(parts))


def main() -> None:
    banner("WP8c : SEMANTIC TRANSITION EXPLANATION")
    print(f"Script version          : {SCRIPT_VERSION}")
    print(f"Project root            : {ROOT}")
    print(f"Master dataset          : {MASTER_PATH}")
    print(f"Semantic dictionary     : {DICTIONARY_PATH}")
    print(f"Contribution analysis   : {CONTRIBUTIONS_PATH}")
    print(f"Rolling window          : {WINDOW_SIZE_YEARS} years")
    print(f"Minimum songs/window    : {MIN_SONGS_PER_WINDOW}")

    require(MASTER_PATH)
    require(DICTIONARY_PATH)
    require(CONTRIBUTIONS_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    banner("LOADING FROZEN MASTER DATASET")
    master = pd.read_parquet(MASTER_PATH)

    required_master_columns = [
        "company",
        "community",
        "analysis_eligible",
    ]
    missing = [
        column
        for column in required_master_columns
        if column not in master.columns
    ]
    if missing:
        raise ValueError(
            "Master dataset is missing required columns: "
            + ", ".join(missing)
        )

    year_column = identify_year_column(master)
    master["analysis_year"] = extract_year(
        master[year_column],
        year_column,
    )

    eligible = master.loc[
        master["analysis_eligible"].eq(True)
    ].copy()

    eligible = eligible.dropna(
        subset=["company", "community", "analysis_year"]
    )

    eligible["company"] = eligible["company"].astype(str).str.strip()
    eligible["community"] = eligible["community"].map(
        normalise_community
    )
    eligible["analysis_year"] = eligible["analysis_year"].astype(int)

    if eligible.empty:
        raise ValueError(
            "No analysis-eligible rows with valid company, community, and year."
        )

    all_communities = sorted(
        eligible["community"].unique(),
        key=natural_key,
    )

    print(f"Master rows             : {len(master):,}")
    print(f"Analysis-eligible rows  : {len(eligible):,}")
    print(f"Year source column      : {year_column}")
    print(
        f"Observed year range     : "
        f"{eligible['analysis_year'].min()}-"
        f"{eligible['analysis_year'].max()}"
    )
    print(f"Companies               : {eligible['company'].nunique():,}")
    print(f"Semantic communities    : {len(all_communities):,}")

    banner("RECONSTRUCTING ROLLING FIVE-YEAR FINGERPRINTS")
    fingerprints, window_inventory = build_rolling_fingerprints(
        eligible,
        all_communities,
    )

    company_community_evidence = build_company_community_evidence(
        eligible,
        max_songs=5,
        max_artists=5,
    )

    print(f"Valid company-windows   : "
          f"{fingerprints[['company', 'window_label']].drop_duplicates().shape[0]:,}")
    print(f"Fingerprint rows        : {len(fingerprints):,}")
    print(f"Excluded sparse windows : "
          f"{int((~window_inventory['valid_window']).sum()):,}")

    banner("LOADING INTERPRETIVE EVIDENCE")
    dictionary = pd.read_csv(
        DICTIONARY_PATH,
        dtype={"community": str},
    )
    dictionary["community"] = dictionary["community"].map(
        normalise_community
    )

    contributions = pd.read_csv(
        CONTRIBUTIONS_PATH,
        dtype={"community": str},
    )
    contributions["community"] = contributions["community"].map(
        normalise_community
    )

    required_dictionary = [
        "community",
        "representative_songs",
        "representative_artists",
        "community_keywords",
    ]
    missing = [
        column
        for column in required_dictionary
        if column not in dictionary.columns
    ]
    if missing:
        raise ValueError(
            "Semantic dictionary is missing required columns: "
            + ", ".join(missing)
        )

    required_contributions = [
        "company",
        "community",
        "contribution_score",
        "contribution_rank",
    ]
    missing = [
        column
        for column in required_contributions
        if column not in contributions.columns
    ]
    if missing:
        raise ValueError(
            "Community contribution output is missing required columns: "
            + ", ".join(missing)
        )

    banner("DECOMPOSING ADJACENT TEMPORAL TRANSITIONS")
    transition_parts: list[pd.DataFrame] = []
    transition_count = 0

    observed_companies = list(pd.unique(fingerprints["company"]))
    company_rank = {
        company: index
        for index, company in enumerate(COMPANY_ORDER + observed_companies)
    }

    for company in sorted(
        observed_companies,
        key=lambda value: company_rank.get(value, 10**9),
    ):
        company_data = fingerprints.loc[
            fingerprints["company"].eq(company)
        ].copy()

        window_table = (
            company_data[
                [
                    "window_start_year",
                    "window_end_year",
                    "window_label",
                    "n_songs_in_window",
                ]
            ]
            .drop_duplicates()
            .sort_values("window_end_year")
            .reset_index(drop=True)
        )

        if len(window_table) < 2:
            continue

        pivot = company_data.pivot_table(
            index="window_end_year",
            columns="community",
            values="probability",
            aggfunc="sum",
            fill_value=0.0,
        ).reindex(
            columns=all_communities,
            fill_value=0.0,
        )

        for position in range(1, len(window_table)):
            previous_row = window_table.iloc[position - 1]
            current_row = window_table.iloc[position]

            previous_end = int(previous_row["window_end_year"])
            current_end = int(current_row["window_end_year"])

            # Only consecutive rolling windows are interpreted as adjacent.
            if current_end - previous_end != 1:
                continue

            p = pivot.loc[previous_end].to_numpy(dtype=float)
            q = pivot.loc[current_end].to_numpy(dtype=float)

            component = js_components(p, q)
            divergence = float(component.sum())
            similarity = cosine_similarity(p, q)

            if divergence > 0:
                js_share = 100.0 * component / divergence
            else:
                js_share = np.zeros_like(component)

            part = pd.DataFrame(
                {
                    "company": company,
                    "transition_index": transition_count + 1,
                    "previous_window": previous_row["window_label"],
                    "current_window": current_row["window_label"],
                    "previous_window_start_year": int(
                        previous_row["window_start_year"]
                    ),
                    "previous_window_end_year": previous_end,
                    "current_window_start_year": int(
                        current_row["window_start_year"]
                    ),
                    "current_window_end_year": current_end,
                    "previous_window_n_songs": int(
                        previous_row["n_songs_in_window"]
                    ),
                    "current_window_n_songs": int(
                        current_row["n_songs_in_window"]
                    ),
                    "community": all_communities,
                    "previous_probability": p,
                    "current_probability": q,
                    "probability_change": q - p,
                    "absolute_probability_change": np.abs(q - p),
                    "js_component": component,
                    "js_share_pct": js_share,
                    "transition_js_divergence": divergence,
                    "transition_js_distance": math.sqrt(
                        max(divergence, 0.0)
                    ),
                    "transition_total_variation": float(
                        0.5 * np.abs(q - p).sum()
                    ),
                    "transition_cosine_similarity": similarity,
                    "transition_cosine_distance": (
                        1.0 - similarity
                        if not pd.isna(similarity)
                        else np.nan
                    ),
                    "transition_euclidean_distance": float(
                        np.linalg.norm(q - p)
                    ),
                }
            )

            part["change_direction"] = np.select(
                [
                    part["probability_change"] > 0,
                    part["probability_change"] < 0,
                ],
                ["Increased", "Decreased"],
                default="Unchanged",
            )

            part["absolute_change_rank"] = (
                part["absolute_probability_change"]
                .rank(method="min", ascending=False)
                .astype("Int64")
            )
            part["js_contribution_rank"] = (
                part["js_component"]
                .rank(method="min", ascending=False)
                .astype("Int64")
            )
            part["increase_rank"] = (
                part["probability_change"]
                .where(part["probability_change"] > 0)
                .rank(method="min", ascending=False)
                .astype("Int64")
            )
            part["decrease_rank"] = (
                part["probability_change"]
                .where(part["probability_change"] < 0)
                .rank(method="min", ascending=True)
                .astype("Int64")
            )

            transition_parts.append(part)
            transition_count += 1

    if not transition_parts:
        raise ValueError(
            "No consecutive rolling-window transitions were constructed."
        )

    transitions = pd.concat(
        transition_parts,
        ignore_index=True,
    )

    dictionary_columns = [
        "community",
        "community_evidence_label",
        "representative_songs",
        "representative_artists",
        "community_keywords",
        "n_songs",
        "n_artists",
    ]
    dictionary_columns = [
        column
        for column in dictionary_columns
        if column in dictionary.columns
    ]

    contribution_columns = [
        "company",
        "community",
        "contribution_score",
        "contribution_rank",
        "organisational_role",
        "enrichment_ratio",
        "log2_enrichment",
        "company_catalogue_share_pct",
        "corpus_share_pct",
    ]
    contribution_columns = [
        column
        for column in contribution_columns
        if column in contributions.columns
    ]

    transitions = transitions.merge(
        dictionary[dictionary_columns],
        on="community",
        how="left",
        validate="many_to_one",
    )

    transitions = transitions.merge(
        company_community_evidence,
        on=["company", "community"],
        how="left",
        validate="many_to_one",
    )

    transitions = transitions.merge(
        contributions[contribution_columns],
        on=["company", "community"],
        how="left",
        validate="many_to_one",
    )

    transitions["transition_id"] = (
        transitions["company"].astype(str)
        + "__"
        + transitions["previous_window"].astype(str)
        + "__to__"
        + transitions["current_window"].astype(str)
    )

    explanations = []
    for transition_id, group in transitions.groupby(
        "transition_id",
        observed=True,
        sort=False,
    ):
        explanations.append(
            {
                "transition_id": transition_id,
                "transition_explanation": normalise_narrative_spacing(
                    transition_explanation(group)
                ),
            }
        )

    explanation_table = pd.DataFrame(explanations)

    transitions = transitions.merge(
        explanation_table,
        on="transition_id",
        how="left",
        validate="many_to_one",
    )

    transitions["_company_order"] = transitions["company"].map(
        {
            company: index
            for index, company in enumerate(COMPANY_ORDER)
        }
    ).fillna(len(COMPANY_ORDER))

    transitions = (
        transitions.sort_values(
            [
                "_company_order",
                "current_window_end_year",
                "js_contribution_rank",
                "community",
            ],
            kind="stable",
        )
        .drop(columns="_company_order")
        .reset_index(drop=True)
    )

    transitions.to_csv(
        TRANSITIONS_PATH,
        index=False,
    )

    banner("BUILDING COMPANY TRANSITION SUMMARY")

    transition_level_records: list[dict[str, object]] = []

    for transition_id, group in transitions.groupby(
        "transition_id",
        observed=True,
        sort=False,
    ):
        first = group.iloc[0]
        maximum_component = float(group["js_component"].max())
        tolerance = max(1e-12, abs(maximum_component) * 1e-9)

        dominant_rows = group.loc[
            np.isclose(
                group["js_component"].to_numpy(dtype=float),
                maximum_component,
                rtol=1e-9,
                atol=tolerance,
            )
        ].sort_values(
            "community",
            key=lambda series: series.map(natural_key),
        )

        dominant_communities = [
            clean_display_text(value)
            for value in dominant_rows["community"].tolist()
        ]

        transition_level_records.append(
            {
                "transition_id": transition_id,
                "company": first["company"],
                "previous_window": first["previous_window"],
                "current_window": first["current_window"],
                "previous_window_end_year": first[
                    "previous_window_end_year"
                ],
                "current_window_end_year": first[
                    "current_window_end_year"
                ],
                "js_divergence": first[
                    "transition_js_divergence"
                ],
                "js_distance": first[
                    "transition_js_distance"
                ],
                "total_variation": first[
                    "transition_total_variation"
                ],
                "cosine_similarity": first[
                    "transition_cosine_similarity"
                ],
                "euclidean_distance": first[
                    "transition_euclidean_distance"
                ],
                "dominant_community": "; ".join(
                    dominant_communities
                ),
                "n_tied_dominant_communities": int(
                    len(dominant_communities)
                ),
                "dominant_community_js_share_pct": float(
                    dominant_rows["js_share_pct"].iloc[0]
                ),
                "transition_explanation": first[
                    "transition_explanation"
                ],
            }
        )

    transition_level = pd.DataFrame(
        transition_level_records
    )

    company_summary = (
        transition_level.groupby(
            "company",
            observed=True,
        )
        .agg(
            n_transitions=("transition_id", "nunique"),
            mean_js_divergence=("js_divergence", "mean"),
            median_js_divergence=("js_divergence", "median"),
            std_js_divergence=("js_divergence", "std"),
            max_js_divergence=("js_divergence", "max"),
            mean_total_variation=("total_variation", "mean"),
            mean_cosine_similarity=("cosine_similarity", "mean"),
            mean_euclidean_distance=("euclidean_distance", "mean"),
            mean_dominant_community_js_share_pct=(
                "dominant_community_js_share_pct",
                "mean",
            ),
        )
        .reset_index()
    )

    largest = (
        transition_level.sort_values(
            ["company", "js_divergence"],
            ascending=[True, False],
            kind="stable",
        )
        .groupby(
            "company",
            observed=True,
        )
        .head(1)
        .set_index("company")
    )

    mappings = {
        "transition_id": "largest_transition_id",
        "previous_window": "largest_transition_previous_window",
        "current_window": "largest_transition_current_window",
        "js_divergence": "largest_transition_js_divergence",
        "dominant_community": (
            "largest_transition_dominant_community"
        ),
        "dominant_community_js_share_pct": (
            "largest_transition_dominant_community_js_share_pct"
        ),
        "transition_explanation": (
            "largest_transition_explanation"
        ),
    }

    for source, target in mappings.items():
        company_summary[target] = (
            company_summary["company"].map(largest[source])
        )

    company_summary["_company_order"] = company_summary["company"].map(
        {
            company: index
            for index, company in enumerate(COMPANY_ORDER)
        }
    ).fillna(len(COMPANY_ORDER))

    company_summary = (
        company_summary.sort_values(
            ["_company_order", "company"],
            kind="stable",
        )
        .drop(columns="_company_order")
        .reset_index(drop=True)
    )

    company_summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    metadata = {
        "script_version": SCRIPT_VERSION,
        "input_master": str(MASTER_PATH),
        "year_source_column": year_column,
        "semantic_dictionary": str(DICTIONARY_PATH),
        "community_contributions": str(CONTRIBUTIONS_PATH),
        "outputs": {
            "semantic_transition_explanations": str(
                TRANSITIONS_PATH
            ),
            "company_transition_summary": str(SUMMARY_PATH),
        },
        "rolling_window_definition": {
            "window_size_years": WINDOW_SIZE_YEARS,
            "minimum_songs_per_window": MIN_SONGS_PER_WINDOW,
            "window_inclusion": (
                "inclusive calendar years from end_year - 4 "
                "through end_year"
            ),
            "transition_definition": (
                "consecutive non-empty five-year rolling windows advanced "
                "in one-year increments"
            ),
        },
        "n_analysis_eligible_rows": int(len(eligible)),
        "n_companies": int(eligible["company"].nunique()),
        "n_communities": int(len(all_communities)),
        "n_valid_company_windows": int(
            fingerprints[
                ["company", "window_label"]
            ].drop_duplicates().shape[0]
        ),
        "n_adjacent_transitions": int(transition_count),
        "method": {
            "probability_change": (
                "current community probability minus previous probability"
            ),
            "js_component": (
                "additive per-community Jensen-Shannon divergence "
                "contribution using natural logarithms"
            ),
            "interpretive_evidence": (
                "company-community representative songs ranked within the "
                "frozen master dataset; global WP8b1 representatives retained "
                "as fallback community evidence"
            ),
            "dominant_tie_handling": (
                "communities with numerically equal maximum JS components are "
                "reported jointly rather than resolved arbitrarily"
            ),
            "manual_topic_labels": False,
        },
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    banner("COMPANY TRANSITION SUMMARY")
    print(
        company_summary[
            [
                "company",
                "n_transitions",
                "mean_js_divergence",
                "max_js_divergence",
                "mean_cosine_similarity",
                "largest_transition_previous_window",
                "largest_transition_current_window",
                "largest_transition_dominant_community",
            ]
        ].to_string(index=False)
    )

    banner("LARGEST SEMANTIC TRANSITION PER COMPANY")
    for row in company_summary.itertuples(index=False):
        print(f"\n{row.company}")
        print(
            normalise_narrative_spacing(
                row.largest_transition_explanation
            )
        )

    banner("OUTPUTS WRITTEN")
    print(TRANSITIONS_PATH)
    print(SUMMARY_PATH)
    print(METADATA_PATH)
    print("\nWP8c completed successfully.")


if __name__ == "__main__":
    main()
