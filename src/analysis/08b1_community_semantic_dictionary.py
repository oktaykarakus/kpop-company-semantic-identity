#!/usr/bin/env python3
"""
WP8b1: Community Semantic Dictionary

Purpose
-------
Create a reproducible semantic lookup table for the 15 graph-derived semantic
communities before WP8c.

The script does not alter the community assignments or the WP8 methodology.
It translates numeric community identifiers into interpretable evidence by
combining:

1. Representative songs:
   Songs closest to their assigned community centroid, using the existing
   `max_centroid_cosine` field. Ties are resolved using weighted degree and
   degree.

2. Representative artists:
   Artists appearing most frequently among the representative songs, followed
   by their total frequency within the community.

3. Community keywords:
   Song-level TF-IDF terms averaged within each community. Lyrics are treated
   as individual documents, binary term occurrence is used to reduce chorus
   repetition effects, and tokens appearing in song titles are removed from
   the corresponding lyric before vectorisation.

The resulting descriptors are evidence-based lexical summaries. They are not
manually imposed topic labels and should be interpreted together with the
representative songs.

Input
-----
outputs/tables/company_master_analysis_v1.parquet

Output
------
outputs/explainability/community_semantic_dictionary.csv
outputs/explainability/community_semantic_dictionary_metadata.json
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


SCRIPT_VERSION = "2026-07-19-v1"

ROOT = Path(__file__).resolve().parents[2]
MASTER_PATH = ROOT / "outputs" / "tables" / "company_master_analysis_v1.parquet"

OUTPUT_DIR = ROOT / "outputs" / "explainability"
DICTIONARY_PATH = OUTPUT_DIR / "community_semantic_dictionary.csv"
METADATA_PATH = OUTPUT_DIR / "community_semantic_dictionary_metadata.json"

N_REPRESENTATIVE_SONGS = 8
N_REPRESENTATIVE_ARTISTS = 5
N_KEYWORDS = 12

ADDITIONAL_STOPWORDS = {
    "yeah", "yea", "oh", "ooh", "uh", "ah", "hey", "woo", "whoa", "woah",
    "la", "na", "da", "dum", "mm", "mmm", "hmm", "ha", "yah", "yo",
    "verse", "chorus", "bridge", "intro", "outro", "prechorus", "refrain",
    "repeat", "repeats",
    "im", "ive", "ill", "id", "youre", "youve", "youll", "youd",
    "hes", "shes", "theyre", "weve", "were", "cant", "dont", "didnt",
    "doesnt", "wont", "wouldnt", "couldnt", "shouldnt", "aint",
    "gonna", "wanna", "gotta",
}


def print_banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")


def natural_community_key(value: object) -> tuple[int, str]:
    text = str(value)
    try:
        return int(float(text)), text
    except ValueError:
        return 10**9, text


def tokenise_title(title: object) -> set[str]:
    if pd.isna(title):
        return set()

    tokens = re.findall(r"[a-z]{3,}", str(title).lower())
    return set(tokens)


def clean_lyrics(lyrics: object, title: object) -> str:
    if pd.isna(lyrics):
        return ""

    text = str(lyrics).lower()

    # Remove common structural annotations.
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = re.sub(
        r"\([^)]*(?:verse|chorus|bridge|intro|outro|refrain)[^)]*\)",
        " ",
        text,
    )

    # Keep English alphabetic content only.
    text = re.sub(r"[^a-z\s']", " ", text)
    text = text.replace("'", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # Remove title tokens from the song's own lyrics to limit title/catchphrase
    # domination. Only tokens of length >= 3 are removed.
    title_tokens = tokenise_title(title)
    if title_tokens:
        text_tokens = [
            token for token in text.split()
            if token not in title_tokens
        ]
        text = " ".join(text_tokens)

    return text


def format_song(title: object, artist: object) -> str:
    title_text = str(title).strip() if not pd.isna(title) else "Untitled"
    artist_text = str(artist).strip() if not pd.isna(artist) else "Unknown artist"
    return f"{title_text} — {artist_text}"


def select_representative_songs(
    community_df: pd.DataFrame,
    n_songs: int,
) -> pd.DataFrame:
    ranked = community_df.copy()

    for column in ["max_centroid_cosine", "weighted_degree", "degree"]:
        ranked[column] = pd.to_numeric(ranked[column], errors="coerce")

    ranked = ranked.sort_values(
        [
            "max_centroid_cosine",
            "weighted_degree",
            "degree",
            "title",
            "artist",
        ],
        ascending=[False, False, False, True, True],
        na_position="last",
        kind="stable",
    )

    return ranked.head(n_songs).copy()


def build_keyword_scores(eligible: pd.DataFrame) -> dict[str, list[tuple[str, float]]]:
    documents = [
        clean_lyrics(lyrics, title)
        for lyrics, title in zip(
            eligible["lyrics"],
            eligible["title"],
        )
    ]

    if not any(documents):
        raise ValueError("No usable lyric text was found for keyword extraction.")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=list(ADDITIONAL_STOPWORDS),
        token_pattern=r"(?u)\b[a-z]{3,}\b",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.80,
        binary=True,
        sublinear_tf=False,
        norm="l2",
    )

    matrix = vectorizer.fit_transform(documents)
    terms = np.asarray(vectorizer.get_feature_names_out())

    keyword_map: dict[str, list[tuple[str, float]]] = {}

    communities = eligible["community"].astype(str).to_numpy()

    for community in sorted(
        eligible["community"].astype(str).unique(),
        key=natural_community_key,
    ):
        mask = communities == community
        community_matrix = matrix[mask]

        if community_matrix.shape[0] == 0:
            keyword_map[community] = []
            continue

        mean_scores = np.asarray(community_matrix.mean(axis=0)).ravel()
        ranked_indices = np.argsort(mean_scores)[::-1]

        ranked_terms: list[tuple[str, float]] = []
        seen_unigrams: set[str] = set()

        for index in ranked_indices:
            score = float(mean_scores[index])
            if score <= 0:
                continue

            term = str(terms[index]).strip()
            if not term:
                continue

            # Avoid near-duplicate unigram/bigram outputs where possible.
            parts = term.split()
            if len(parts) == 1:
                if term in seen_unigrams:
                    continue
                seen_unigrams.add(term)

            ranked_terms.append((term, score))

            if len(ranked_terms) >= N_KEYWORDS:
                break

        keyword_map[community] = ranked_terms

    return keyword_map


def build_dictionary(eligible: pd.DataFrame) -> pd.DataFrame:
    keyword_map = build_keyword_scores(eligible)
    records: list[dict[str, object]] = []

    for community in sorted(
        eligible["community"].astype(str).unique(),
        key=natural_community_key,
    ):
        group = eligible.loc[
            eligible["community"].astype(str).eq(community)
        ].copy()

        representatives = select_representative_songs(
            group,
            N_REPRESENTATIVE_SONGS,
        )

        representative_song_strings = [
            format_song(row.title, row.artist)
            for row in representatives.itertuples(index=False)
        ]

        representative_artist_counter = Counter(
            str(value).strip()
            for value in representatives["artist"].dropna()
            if str(value).strip()
        )
        full_artist_counter = Counter(
            str(value).strip()
            for value in group["artist"].dropna()
            if str(value).strip()
        )

        artist_candidates = sorted(
            full_artist_counter,
            key=lambda artist: (
                -representative_artist_counter.get(artist, 0),
                -full_artist_counter.get(artist, 0),
                artist.lower(),
            ),
        )
        representative_artists = artist_candidates[:N_REPRESENTATIVE_ARTISTS]

        keyword_pairs = keyword_map.get(community, [])
        keywords = [term for term, _ in keyword_pairs]
        keyword_scores = [score for _, score in keyword_pairs]

        # A compact evidence label for use in later procedural scripts.
        label_terms = keywords[:4]
        evidence_label = (
            " / ".join(term.title() for term in label_terms)
            if label_terms
            else f"Community {community}"
        )

        record: dict[str, object] = {
            "community": community,
            "community_evidence_label": evidence_label,
            "n_songs": int(len(group)),
            "n_artists": int(group["artist_normalised"].nunique()),
            "n_companies": int(group["company"].nunique()),
            "community_keywords": "; ".join(keywords),
            "representative_songs": "; ".join(representative_song_strings),
            "representative_artists": "; ".join(representative_artists),
            "mean_centroid_similarity": pd.to_numeric(
                group["max_centroid_cosine"],
                errors="coerce",
            ).mean(),
            "median_centroid_similarity": pd.to_numeric(
                group["max_centroid_cosine"],
                errors="coerce",
            ).median(),
            "mean_weighted_degree": pd.to_numeric(
                group["weighted_degree"],
                errors="coerce",
            ).mean(),
            "mean_participation_coefficient": pd.to_numeric(
                group["participation_coefficient"],
                errors="coerce",
            ).mean(),
            "mean_bridge_affinity": pd.to_numeric(
                group["bridge_affinity_score"],
                errors="coerce",
            ).mean(),
            "mean_local_novelty": pd.to_numeric(
                group["local_novelty"],
                errors="coerce",
            ).mean(),
            "mean_centroid_novelty": pd.to_numeric(
                group["centroid_novelty"],
                errors="coerce",
            ).mean(),
        }

        for rank in range(N_REPRESENTATIVE_SONGS):
            position = rank + 1
            if rank < len(representatives):
                row = representatives.iloc[rank]
                record[f"representative_song_{position}"] = row["title"]
                record[f"representative_artist_{position}"] = row["artist"]
                record[f"representative_song_{position}_centroid_similarity"] = (
                    pd.to_numeric(
                        pd.Series([row["max_centroid_cosine"]]),
                        errors="coerce",
                    ).iloc[0]
                )
            else:
                record[f"representative_song_{position}"] = ""
                record[f"representative_artist_{position}"] = ""
                record[f"representative_song_{position}_centroid_similarity"] = np.nan

        for rank in range(N_KEYWORDS):
            position = rank + 1
            if rank < len(keyword_pairs):
                term, score = keyword_pairs[rank]
                record[f"keyword_{position}"] = term
                record[f"keyword_{position}_score"] = score
            else:
                record[f"keyword_{position}"] = ""
                record[f"keyword_{position}_score"] = np.nan

        records.append(record)

    dictionary = pd.DataFrame(records)
    dictionary["_community_sort"] = dictionary["community"].map(
        lambda value: natural_community_key(value)[0]
    )
    dictionary = (
        dictionary.sort_values(
            ["_community_sort", "community"],
            kind="stable",
        )
        .drop(columns="_community_sort")
        .reset_index(drop=True)
    )

    return dictionary


def main() -> None:
    print_banner("WP8b1 : COMMUNITY SEMANTIC DICTIONARY")
    print(f"Script version          : {SCRIPT_VERSION}")
    print(f"Project root            : {ROOT}")
    print(f"Master dataset          : {MASTER_PATH}")
    print(f"Output directory        : {OUTPUT_DIR}")

    require_file(MASTER_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print_banner("LOADING EXISTING PIPELINE OUTPUT")
    master = pd.read_parquet(MASTER_PATH)

    required_columns = [
        "title",
        "artist",
        "artist_normalised",
        "company",
        "community",
        "lyrics",
        "analysis_eligible",
        "max_centroid_cosine",
        "weighted_degree",
        "degree",
        "participation_coefficient",
        "bridge_affinity_score",
        "local_novelty",
        "centroid_novelty",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in master.columns
    ]
    if missing_columns:
        raise ValueError(
            "Master dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    eligible = master.loc[master["analysis_eligible"].eq(True)].copy()
    eligible = eligible.dropna(
        subset=["company", "community", "title", "artist"]
    )

    if eligible.empty:
        raise ValueError("No analysis-eligible rows are available.")

    eligible["community"] = eligible["community"].astype(str)

    print(f"Master rows             : {len(master):,}")
    print(f"Analysis-eligible rows  : {len(eligible):,}")
    print(f"Companies               : {eligible['company'].nunique():,}")
    print(f"Semantic communities    : {eligible['community'].nunique():,}")

    print_banner("BUILDING COMMUNITY SEMANTIC DICTIONARY")
    dictionary = build_dictionary(eligible)
    dictionary.to_csv(DICTIONARY_PATH, index=False)

    metadata = {
        "script_version": SCRIPT_VERSION,
        "input": str(MASTER_PATH),
        "output": str(DICTIONARY_PATH),
        "analysis_eligible_rows": int(len(eligible)),
        "n_communities": int(eligible["community"].nunique()),
        "representative_song_method": {
            "primary_sort": "max_centroid_cosine descending",
            "tie_breakers": [
                "weighted_degree descending",
                "degree descending",
                "title ascending",
                "artist ascending",
            ],
            "n_representative_songs": N_REPRESENTATIVE_SONGS,
        },
        "keyword_method": {
            "document_unit": "individual song lyrics",
            "aggregation": "mean song-level TF-IDF within community",
            "binary_term_occurrence": True,
            "ngram_range": [1, 2],
            "minimum_document_frequency": 3,
            "maximum_document_frequency": 0.80,
            "song_title_tokens_removed_from_own_lyrics": True,
            "n_keywords": N_KEYWORDS,
        },
        "interpretation": (
            "Evidence labels and keywords are reproducible lexical descriptors. "
            "They are not manually assigned topic names."
        ),
    }

    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(
        dictionary[
            [
                "community",
                "community_evidence_label",
                "n_songs",
                "n_artists",
                "representative_songs",
                "community_keywords",
            ]
        ].to_string(index=False)
    )

    print_banner("OUTPUTS WRITTEN")
    print(DICTIONARY_PATH)
    print(METADATA_PATH)
    print("\nWP8b1 completed successfully.")


if __name__ == "__main__":
    main()
