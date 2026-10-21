#!/usr/bin/env python3
"""
WP8b: Community Contribution Analysis

Scientific question
-------------------
Which semantic communities contribute most strongly to each entertainment
company's organisational identity?

This version extends the original WP8b analysis by attaching reproducible,
corpus-derived semantic descriptors to every community. It does not introduce
a new work package or new output file. The descriptors are written directly
into the two established WP8b outputs so that WP8c and later analyses can refer
to interpretable themes rather than numeric community identifiers.

Community descriptors
---------------------
Each semantic community is represented as a single aggregated document built
from all analysis-eligible lyrics assigned to that community. TF-IDF is then
calculated across the 15 community documents.

For every community, the script records:
    - community_label: the three most distinctive terms
    - community_keywords: the ten most distinctive terms

These are descriptive lexical signatures rather than manually imposed topic
names. They preserve reproducibility and avoid introducing subjective labels.

Community contribution analysis
-------------------------------
For every company-community pair, the script calculates:
    - company community probability
    - corpus community probability
    - enrichment difference
    - enrichment ratio
    - log2 enrichment
    - signed contribution score
    - absolute contribution score
    - within-company rankings
    - qualitative contribution category

Contribution score:

    company_probability * log2(company_probability / corpus_probability)

Positive values identify overrepresented communities that help define a
company. Negative values identify underrepresented communities that distinguish
a company through relative absence.

Inputs
------
outputs/tables/company_master_analysis_v1.parquet

Outputs
-------
outputs/explainability/community_contributions.csv
outputs/explainability/community_rankings.csv
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


SCRIPT_VERSION = "2026-07-19-v2"

ROOT = Path(__file__).resolve().parents[2]

MASTER_PATH = ROOT / "outputs" / "tables" / "company_master_analysis_v1.parquet"

OUTPUT_DIR = ROOT / "outputs" / "explainability"
CONTRIBUTIONS_PATH = OUTPUT_DIR / "community_contributions.csv"
RANKINGS_PATH = OUTPUT_DIR / "community_rankings.csv"
METADATA_PATH = OUTPUT_DIR / "community_contribution_metadata.json"

COMPANY_ORDER = ["HYBE", "JYP", "SM", "YG"]

ADDITIONAL_STOPWORDS = {
    "yeah", "yea", "oh", "ooh", "uh", "ah", "hey", "woo", "whoa", "woah",
    "la", "na", "da", "dum", "mm", "mmm", "hmm", "ha", "ya", "yo",
    "baby", "babe", "girl", "boy",
    "verse", "chorus", "bridge", "intro", "outro", "prechorus", "refrain",
    "repeat", "repeats",
    "im", "ive", "ill", "id", "youre", "youve", "youll", "youd",
    "hes", "shes", "theyre", "weve", "were", "cant", "dont", "didnt",
    "doesnt", "wont", "wouldnt", "couldnt", "shouldnt", "aint",
}


def print_banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")


def normalise_lyrics(text: object) -> str:
    if pd.isna(text):
        return ""

    cleaned = str(text).lower()
    cleaned = re.sub(r"\[[^\]]+\]", " ", cleaned)
    cleaned = re.sub(r"\([^)]*(?:verse|chorus|bridge|intro|outro)[^)]*\)", " ", cleaned)
    cleaned = re.sub(r"[^a-z\s']", " ", cleaned)
    cleaned = cleaned.replace("'", "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def build_community_descriptors(
    eligible: pd.DataFrame,
) -> pd.DataFrame:
    community_documents = (
        eligible.assign(clean_lyrics=eligible["lyrics"].map(normalise_lyrics))
        .groupby("community", observed=True)["clean_lyrics"]
        .apply(lambda values: " ".join(value for value in values if value))
        .reset_index()
    )

    if community_documents["clean_lyrics"].str.len().eq(0).all():
        raise ValueError("No usable lyric text was found for community labelling.")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=list(ADDITIONAL_STOPWORDS),
        token_pattern=r"(?u)\b[a-z]{3,}\b",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.90,
        sublinear_tf=True,
        norm="l2",
    )

    matrix = vectorizer.fit_transform(community_documents["clean_lyrics"])
    terms = np.asarray(vectorizer.get_feature_names_out())

    records: list[dict[str, object]] = []

    for row_index, row in community_documents.iterrows():
        scores = matrix.getrow(row_index).toarray().ravel()
        ranked_indices = np.argsort(scores)[::-1]
        ranked_indices = [
            index for index in ranked_indices if scores[index] > 0
        ]

        top_terms = terms[ranked_indices[:10]].tolist()
        label_terms = top_terms[:3]

        if not top_terms:
            community_label = f"Community {row['community']}"
            community_keywords = ""
        else:
            community_label = " / ".join(
                term.title() for term in label_terms
            )
            community_keywords = "; ".join(top_terms)

        records.append(
            {
                "community": str(row["community"]),
                "community_label": community_label,
                "community_keywords": community_keywords,
                "descriptor_term_1": top_terms[0] if len(top_terms) > 0 else "",
                "descriptor_term_2": top_terms[1] if len(top_terms) > 1 else "",
                "descriptor_term_3": top_terms[2] if len(top_terms) > 2 else "",
                "n_descriptor_terms": len(top_terms),
            }
        )

    descriptors = pd.DataFrame(records)
    return descriptors.sort_values("community").reset_index(drop=True)


def qualitative_contribution(percentile: float) -> str:
    if pd.isna(percentile):
        return "Unavailable"
    if percentile <= 20:
        return "Very Low"
    if percentile <= 40:
        return "Low"
    if percentile <= 60:
        return "Moderate"
    if percentile <= 80:
        return "High"
    return "Very High"


def enrichment_category(log2_enrichment: float) -> str:
    if pd.isna(log2_enrichment):
        return "Unavailable"
    if log2_enrichment >= 1.0:
        return "Strongly Enriched"
    if log2_enrichment >= 0.25:
        return "Enriched"
    if log2_enrichment > -0.25:
        return "Near Corpus Average"
    if log2_enrichment > -1.0:
        return "Underrepresented"
    return "Strongly Underrepresented"


def contribution_direction(score: float) -> str:
    if pd.isna(score):
        return "Unavailable"
    if score > 0:
        return "Identity Defining"
    if score < 0:
        return "Identity Contrasting"
    return "Neutral"


def main() -> None:
    print_banner("WP8b : COMMUNITY CONTRIBUTION ANALYSIS")
    print(f"Script version          : {SCRIPT_VERSION}")
    print(f"Project root            : {ROOT}")
    print(f"Master dataset          : {MASTER_PATH}")
    print(f"Output directory        : {OUTPUT_DIR}")

    require_file(MASTER_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print_banner("LOADING EXISTING PIPELINE OUTPUT")
    master = pd.read_parquet(MASTER_PATH)

    required_columns = ["company", "community", "analysis_eligible", "lyrics"]
    missing_columns = [
        column for column in required_columns if column not in master.columns
    ]
    if missing_columns:
        raise ValueError(
            "Master dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    eligible = master.loc[master["analysis_eligible"].eq(True)].copy()
    eligible = eligible.dropna(subset=["company", "community"])

    if eligible.empty:
        raise ValueError("No analysis-eligible rows are available.")

    eligible["company"] = eligible["company"].astype(str)
    eligible["community"] = eligible["community"].astype(str)

    observed_companies = set(eligible["company"].unique())
    expected_companies = set(COMPANY_ORDER)
    if observed_companies != expected_companies:
        raise ValueError(
            "Unexpected company set. "
            f"Expected {sorted(expected_companies)}, "
            f"observed {sorted(observed_companies)}."
        )

    print(f"Master rows             : {len(master):,}")
    print(f"Analysis-eligible rows  : {len(eligible):,}")
    print(f"Companies               : {eligible['company'].nunique():,}")
    print(f"Semantic communities    : {eligible['community'].nunique():,}")

    print_banner("DERIVING COMMUNITY SEMANTIC DESCRIPTORS")
    descriptors = build_community_descriptors(eligible)

    print(
        descriptors[
            ["community", "community_label", "community_keywords"]
        ].to_string(index=False)
    )

    print_banner("CALCULATING COMMUNITY CONTRIBUTIONS")

    company_totals = (
        eligible.groupby("company", observed=True)
        .size()
        .rename("company_song_count")
        .reset_index()
    )

    corpus_total = int(len(eligible))

    corpus_counts = (
        eligible.groupby("community", observed=True)
        .size()
        .rename("corpus_community_count")
        .reset_index()
    )
    corpus_counts["corpus_community_probability"] = (
        corpus_counts["corpus_community_count"] / corpus_total
    )

    all_pairs = pd.MultiIndex.from_product(
        [COMPANY_ORDER, sorted(eligible["community"].unique())],
        names=["company", "community"],
    ).to_frame(index=False)

    company_counts = (
        eligible.groupby(["company", "community"], observed=True)
        .size()
        .rename("company_community_count")
        .reset_index()
    )

    contributions = (
        all_pairs.merge(
            company_counts,
            on=["company", "community"],
            how="left",
        )
        .merge(
            company_totals,
            on="company",
            how="left",
        )
        .merge(
            corpus_counts,
            on="community",
            how="left",
        )
        .merge(
            descriptors,
            on="community",
            how="left",
        )
    )

    contributions["company_community_count"] = (
        contributions["company_community_count"].fillna(0).astype(int)
    )

    contributions["company_community_probability"] = (
        contributions["company_community_count"]
        / contributions["company_song_count"]
    )

    contributions["enrichment_difference"] = (
        contributions["company_community_probability"]
        - contributions["corpus_community_probability"]
    )

    contributions["enrichment_ratio"] = np.divide(
        contributions["company_community_probability"],
        contributions["corpus_community_probability"],
        out=np.full(len(contributions), np.nan, dtype=float),
        where=contributions["corpus_community_probability"].to_numpy() > 0,
    )

    positive_ratio = contributions["enrichment_ratio"].to_numpy() > 0
    log2_enrichment = np.full(len(contributions), np.nan, dtype=float)
    log2_enrichment[positive_ratio] = np.log2(
        contributions.loc[positive_ratio, "enrichment_ratio"].to_numpy()
    )
    contributions["log2_enrichment"] = log2_enrichment

    contributions["contribution_score"] = np.where(
        contributions["company_community_probability"] > 0,
        contributions["company_community_probability"]
        * contributions["log2_enrichment"],
        0.0,
    )

    contributions["absolute_contribution_score"] = contributions[
        "contribution_score"
    ].abs()

    contributions["positive_contribution_score"] = contributions[
        "contribution_score"
    ].clip(lower=0)

    contributions["negative_contribution_score"] = contributions[
        "contribution_score"
    ].clip(upper=0)

    contributions["contribution_rank"] = (
        contributions.groupby("company", observed=True)["contribution_score"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )

    contributions["absolute_contribution_rank"] = (
        contributions.groupby("company", observed=True)[
            "absolute_contribution_score"
        ]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )

    contributions["enrichment_rank"] = (
        contributions.groupby("company", observed=True)["log2_enrichment"]
        .rank(method="min", ascending=False, na_option="bottom")
        .astype("Int64")
    )

    contributions["company_probability_rank"] = (
        contributions.groupby("company", observed=True)[
            "company_community_probability"
        ]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )

    contributions["contribution_percentile"] = (
        contributions.groupby("company", observed=True)["contribution_score"]
        .rank(method="average", pct=True)
        * 100.0
    )

    contributions["contribution_category"] = contributions[
        "contribution_percentile"
    ].map(qualitative_contribution)

    contributions["enrichment_category"] = contributions[
        "log2_enrichment"
    ].map(enrichment_category)

    contributions["organisational_role"] = contributions[
        "contribution_score"
    ].map(contribution_direction)

    contributions["company_catalogue_share_pct"] = (
        contributions["company_community_probability"] * 100.0
    )
    contributions["corpus_share_pct"] = (
        contributions["corpus_community_probability"] * 100.0
    )

    company_order_map = {
        company: index for index, company in enumerate(COMPANY_ORDER)
    }
    contributions["_company_order"] = contributions["company"].map(
        company_order_map
    )

    contribution_columns = [
        "company",
        "community",
        "community_label",
        "community_keywords",
        "company_community_count",
        "company_song_count",
        "company_community_probability",
        "company_catalogue_share_pct",
        "corpus_community_count",
        "corpus_community_probability",
        "corpus_share_pct",
        "enrichment_difference",
        "enrichment_ratio",
        "log2_enrichment",
        "contribution_score",
        "absolute_contribution_score",
        "positive_contribution_score",
        "negative_contribution_score",
        "company_probability_rank",
        "enrichment_rank",
        "contribution_rank",
        "absolute_contribution_rank",
        "contribution_percentile",
        "contribution_category",
        "enrichment_category",
        "organisational_role",
    ]

    contributions_output = (
        contributions.sort_values(
            ["_company_order", "contribution_rank", "community"],
            kind="stable",
        )[contribution_columns]
        .reset_index(drop=True)
    )

    rankings = contributions.loc[
        contributions["company_community_count"].gt(0)
    ].copy()

    rankings["identity_defining_rank"] = (
        rankings.groupby("company", observed=True)[
            "positive_contribution_score"
        ]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )

    rankings["identity_contrasting_rank"] = (
        rankings.groupby("company", observed=True)[
            "negative_contribution_score"
        ]
        .rank(method="min", ascending=True)
        .astype("Int64")
    )

    ranking_columns = [
        "company",
        "community",
        "community_label",
        "community_keywords",
        "company_community_probability",
        "corpus_community_probability",
        "enrichment_ratio",
        "log2_enrichment",
        "contribution_score",
        "identity_defining_rank",
        "identity_contrasting_rank",
        "company_probability_rank",
        "enrichment_rank",
        "enrichment_category",
        "organisational_role",
    ]

    rankings_output = (
        rankings.sort_values(
            ["_company_order", "identity_defining_rank", "community"],
            kind="stable",
        )[ranking_columns]
        .reset_index(drop=True)
    )

    contributions_output.to_csv(CONTRIBUTIONS_PATH, index=False)
    rankings_output.to_csv(RANKINGS_PATH, index=False)

    metadata = {
        "script_version": SCRIPT_VERSION,
        "input": str(MASTER_PATH),
        "outputs": {
            "community_contributions": str(CONTRIBUTIONS_PATH),
            "community_rankings": str(RANKINGS_PATH),
        },
        "analysis_eligible_rows": int(len(eligible)),
        "n_companies": int(eligible["company"].nunique()),
        "n_communities": int(eligible["community"].nunique()),
        "community_descriptor_method": {
            "document_unit": "all eligible lyrics aggregated by community",
            "vectorisation": "TF-IDF across community documents",
            "ngram_range": [1, 2],
            "label_terms": 3,
            "keyword_terms": 10,
            "interpretation": (
                "descriptive lexical signatures, not manually imposed topics"
            ),
        },
        "company_probability_definition": (
            "company-community song count divided by company song count"
        ),
        "corpus_probability_definition": (
            "corpus-community song count divided by total eligible song count"
        ),
        "enrichment_ratio_definition": (
            "company community probability divided by corpus community probability"
        ),
        "contribution_score_definition": (
            "company_probability * log2(company_probability / corpus_probability)"
        ),
        "interpretation": {
            "positive_score": (
                "community is overrepresented and contributes positively to "
                "the company's distinctive semantic identity"
            ),
            "negative_score": (
                "community is underrepresented and helps distinguish the "
                "company by relative absence"
            ),
            "zero_score": (
                "community is absent or represented approximately at the "
                "corpus expectation"
            ),
        },
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print_banner("TOP IDENTITY-DEFINING COMMUNITIES")
    top_positive = contributions.loc[
        contributions["contribution_score"].gt(0)
    ].copy()
    top_positive = (
        top_positive.sort_values(
            ["_company_order", "contribution_rank", "community"],
            kind="stable",
        )
        .groupby("company", observed=True)
        .head(5)
    )

    print(
        top_positive[
            [
                "company",
                "community",
                "community_label",
                "company_catalogue_share_pct",
                "corpus_share_pct",
                "enrichment_ratio",
                "contribution_score",
                "contribution_rank",
            ]
        ].to_string(index=False)
    )

    print_banner("STRONGEST IDENTITY-CONTRASTING COMMUNITIES")
    top_negative = contributions.loc[
        contributions["contribution_score"].lt(0)
    ].copy()
    top_negative = (
        top_negative.sort_values(
            ["_company_order", "contribution_score", "community"],
            ascending=[True, True, True],
            kind="stable",
        )
        .groupby("company", observed=True)
        .head(3)
    )

    print(
        top_negative[
            [
                "company",
                "community",
                "community_label",
                "company_catalogue_share_pct",
                "corpus_share_pct",
                "enrichment_ratio",
                "contribution_score",
            ]
        ].to_string(index=False)
    )

    print_banner("OUTPUTS WRITTEN")
    print(CONTRIBUTIONS_PATH)
    print(RANKINGS_PATH)
    print(METADATA_PATH)

    print("\nWP8b completed successfully.")


if __name__ == "__main__":
    main()
