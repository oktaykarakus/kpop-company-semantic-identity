#!/usr/bin/env python3
"""
WP8f: Integrated Organisational Semantic Identity

Purpose
-------
Synthesise WP8a-WP8e into a traceable, publication-ready organisational
semantic identity for each company.

This script performs no new statistical inference. It introduces no new
latent variables, distance measures, or hypothesis tests. Every output claim
is derived from previously computed WP8 results.

Inputs
------
outputs/explainability/company_signature_summary.csv
outputs/explainability/company_signature_rankings.csv
outputs/explainability/community_rankings.csv
outputs/explainability/community_semantic_dictionary.csv
outputs/explainability/company_transition_summary.csv
outputs/explainability/organisational_semantic_archetypes.csv
outputs/explainability/company_differentiation_rankings.csv

Outputs
-------
outputs/explainability/integrated_company_identity.csv
outputs/explainability/company_identity_summary.csv
outputs/explainability/company_identity_narratives.csv
outputs/explainability/identity_evidence_traceability.csv
outputs/explainability/cross_analysis_consistency.csv
outputs/explainability/integrated_identity_metadata.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SCRIPT_VERSION = "2026-08-01-v4-wp5-aligned"


def find_project_root(start: Path) -> Path:
    candidates = [start.resolve(), Path.cwd().resolve()]
    for candidate in candidates:
        for parent in [candidate, *candidate.parents]:
            if (parent / "outputs" / "tables").exists():
                return parent
    raise FileNotFoundError(
        "Could not locate the project root containing outputs/tables."
    )


ROOT = find_project_root(Path(__file__).parent)
OUT = ROOT / "outputs" / "explainability"

SIGNATURE_SUMMARY_PATH = OUT / "company_signature_summary.csv"
SIGNATURE_RANKINGS_PATH = OUT / "company_signature_rankings.csv"
COMMUNITY_RANKINGS_PATH = OUT / "community_rankings.csv"
COMMUNITY_DICTIONARY_PATH = OUT / "community_semantic_dictionary.csv"
TRANSITION_SUMMARY_PATH = OUT / "company_transition_summary.csv"
TEMPORAL_STABILITY_PATH = (
    ROOT / "outputs" / "temporal" / "rolling5"
    / "temporal_stability_summary.csv"
)
ARCHETYPES_PATH = OUT / "organisational_semantic_archetypes.csv"
DIFFERENTIATION_PATH = OUT / "company_differentiation_rankings.csv"

INTEGRATED_PATH = OUT / "integrated_company_identity.csv"
SUMMARY_PATH = OUT / "company_identity_summary.csv"
NARRATIVES_PATH = OUT / "company_identity_narratives.csv"
NARRATIVES_TEXT_PATH = OUT / "company_identity_narratives.txt"
TRACEABILITY_PATH = OUT / "identity_evidence_traceability.csv"
CONSISTENCY_PATH = OUT / "cross_analysis_consistency.csv"
METADATA_PATH = OUT / "integrated_identity_metadata.json"


def section(title: str) -> None:
    line = "=" * 78
    print(f"\n{line}\n{title}\n{line}")


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")


def clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return " ".join(str(value).split())


def normalise_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", clean_text(value).lower())


def first_existing_column(
    frame: pd.DataFrame,
    candidates: Iterable[str],
    *,
    required: bool = False,
    label: str = "column",
) -> str | None:
    lookup = {normalise_name(column): column for column in frame.columns}
    for candidate in candidates:
        key = normalise_name(candidate)
        if key in lookup:
            return lookup[key]

    if required:
        raise KeyError(
            f"Could not resolve {label}. Available columns: "
            + ", ".join(map(str, frame.columns))
        )
    return None


def numeric_series(
    frame: pd.DataFrame,
    candidates: Iterable[str],
    *,
    required: bool = False,
    label: str = "numeric field",
) -> pd.Series:
    column = first_existing_column(
        frame,
        candidates,
        required=required,
        label=label,
    )
    if column is None:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def text_series(
    frame: pd.DataFrame,
    candidates: Iterable[str],
    *,
    required: bool = False,
    label: str = "text field",
) -> pd.Series:
    column = first_existing_column(
        frame,
        candidates,
        required=required,
        label=label,
    )
    if column is None:
        return pd.Series("", index=frame.index, dtype=object)
    return frame[column].map(clean_text)


def load_csv(path: Path) -> pd.DataFrame:
    require(path)
    return pd.read_csv(path)


def resolve_company(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    company_column = first_existing_column(
        frame,
        ["company", "label", "organisation", "organization"],
        required=True,
        label=f"company identifier in {source_name}",
    )
    resolved = frame.copy()
    resolved["company"] = resolved[company_column].map(clean_text)
    resolved = resolved[resolved["company"].ne("")].copy()
    return resolved


def resolve_signature(
    summary: pd.DataFrame,
    rankings: pd.DataFrame,
) -> pd.DataFrame:
    summary = resolve_company(summary, "company_signature_summary.csv")
    rankings = resolve_company(rankings, "company_signature_rankings.csv")

    result = pd.DataFrame({"company": sorted(summary["company"].unique())})

    strength_candidates = [
        "signature_strength",
        "organisational_distinctiveness",
        "organizational_distinctiveness",
        "distinctiveness_score",
        "relative_distinctiveness",
    ]
    rank_candidates = [
        "signature_rank",
        "distinctiveness_rank",
        "rank",
        "overall_rank",
    ]

    summary_strength = numeric_series(
        summary,
        strength_candidates,
        label="signature strength",
    )
    summary_rank = numeric_series(
        summary,
        rank_candidates,
        label="signature rank",
    )

    summary_core = pd.DataFrame(
        {
            "company": summary["company"],
            "signature_strength": summary_strength,
            "signature_rank": summary_rank,
        }
    ).drop_duplicates("company")

    ranking_strength = numeric_series(
        rankings,
        strength_candidates,
        label="signature strength",
    )
    ranking_rank = numeric_series(
        rankings,
        rank_candidates,
        label="signature rank",
    )

    ranking_core = pd.DataFrame(
        {
            "company": rankings["company"],
            "signature_strength_rankings": ranking_strength,
            "signature_rank_rankings": ranking_rank,
        }
    ).drop_duplicates("company")

    result = result.merge(summary_core, on="company", how="left")
    result = result.merge(ranking_core, on="company", how="left")

    result["signature_strength"] = result["signature_strength"].fillna(
        result["signature_strength_rankings"]
    )
    result["signature_rank"] = result["signature_rank"].fillna(
        result["signature_rank_rankings"]
    )

    if result["signature_rank"].isna().all():
        result["signature_rank"] = (
            result["signature_strength"]
            .rank(method="min", ascending=False)
        )

    result["signature_rank"] = pd.to_numeric(
        result["signature_rank"],
        errors="coerce",
    ).astype("Int64")

    return result[
        [
            "company",
            "signature_strength",
            "signature_rank",
        ]
    ]


def resolve_archetypes(frame: pd.DataFrame) -> pd.DataFrame:
    frame = resolve_company(
        frame,
        "organisational_semantic_archetypes.csv",
    )

    result = pd.DataFrame({"company": frame["company"]})
    result["semantic_archetype"] = text_series(
        frame,
        ["semantic_archetype", "archetype"],
        required=True,
        label="semantic archetype",
    )
    result["exploration_score"] = numeric_series(
        frame,
        ["exploration_score"],
        required=True,
        label="exploration score",
    )
    result["integration_score"] = numeric_series(
        frame,
        ["integration_score"],
        required=True,
        label="integration score",
    )
    result["breadth_score"] = numeric_series(
        frame,
        ["breadth_score"],
        required=True,
        label="breadth score",
    )
    result["adaptation_score"] = numeric_series(
        frame,
        ["adaptation_score"],
        required=True,
        label="adaptation score",
    )
    result["dominant_dimension"] = text_series(
        frame,
        ["dominant_dimension"],
    )
    result["positive_dominant_dimensions"] = text_series(
        frame,
        ["positive_dominant_dimensions"],
    )
    result["dominance_type"] = text_series(
        frame,
        ["dominance_type"],
    )
    result["semantic_fingerprint"] = text_series(
        frame,
        ["semantic_fingerprint", "semantic_fingerprint_vector"],
    )
    result["archetype_interpretation"] = text_series(
        frame,
        [
            "strategic_interpretation",
            "archetype_interpretation",
            "interpretation",
        ],
    )

    return result.drop_duplicates("company")


def resolve_transitions(frame: pd.DataFrame) -> pd.DataFrame:
    frame = resolve_company(frame, "company_transition_summary.csv")

    result = pd.DataFrame({"company": frame["company"]})
    result["transition_count"] = numeric_series(
        frame,
        [
            "transition_count",
            "n_transitions",
            "number_of_transitions",
            "transitions",
        ],
    )
    result["mean_transition_js"] = numeric_series(
        frame,
        [
            "mean_js",
            "mean_js_divergence",
            "mean_transition_js",
            "average_js_divergence",
        ],
    )
    result["max_transition_js"] = numeric_series(
        frame,
        [
            "max_js",
            "max_js_divergence",
            "maximum_js_divergence",
        ],
    )
    result["mean_transition_cosine"] = numeric_series(
        frame,
        [
            "mean_cosine",
            "mean_cosine_similarity",
            "cosine_similarity",
            "mean_transition_cosine",
        ],
    )
    result["temporal_interpretation"] = text_series(
        frame,
        [
            "temporal_interpretation",
            "transition_interpretation",
            "interpretation",
            "summary",
        ],
    )

    return result.drop_duplicates("company")


def resolve_stability(frame: pd.DataFrame) -> pd.DataFrame:
    """Resolve the frozen WP5 temporal-stability evidence.

    WP5 is the temporal source of truth for the manuscript. This resolver
    intentionally avoids reconstructing temporal magnitudes inside WP8.
    """
    frame = resolve_company(
        frame,
        "temporal_stability_summary.csv",
    )

    required = [
        "n_transitions",
        "mean_js",
        "median_js",
        "std_js",
        "max_js",
        "mean_cosine",
        "n_change_points",
        "change_point_rate",
        "stability_index",
        "stability_rank",
        "stability_category",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            "WP5 temporal stability output is missing required columns: "
            + ", ".join(missing)
        )

    result = pd.DataFrame({"company": frame["company"]})
    result["transition_count"] = pd.to_numeric(
        frame["n_transitions"], errors="coerce"
    ).astype("Int64")
    result["mean_transition_js"] = pd.to_numeric(
        frame["mean_js"], errors="coerce"
    )
    result["median_transition_js"] = pd.to_numeric(
        frame["median_js"], errors="coerce"
    )
    result["std_transition_js"] = pd.to_numeric(
        frame["std_js"], errors="coerce"
    )
    result["max_transition_js"] = pd.to_numeric(
        frame["max_js"], errors="coerce"
    )
    result["mean_transition_cosine"] = pd.to_numeric(
        frame["mean_cosine"], errors="coerce"
    )
    result["n_change_points"] = pd.to_numeric(
        frame["n_change_points"], errors="coerce"
    ).astype("Int64")
    result["change_point_rate"] = pd.to_numeric(
        frame["change_point_rate"], errors="coerce"
    )
    result["stability_index"] = pd.to_numeric(
        frame["stability_index"], errors="coerce"
    )
    result["stability_rank"] = pd.to_numeric(
        frame["stability_rank"], errors="coerce"
    ).astype("Int64")
    result["stability_category"] = frame["stability_category"].map(clean_text)
    result["largest_change_transition"] = text_series(
        frame, ["largest_change_transition"]
    )
    result["largest_change_js"] = numeric_series(
        frame, ["largest_change_js"]
    )
    result["largest_change_severity"] = text_series(
        frame, ["largest_change_severity"]
    )

    result["temporal_interpretation"] = result.apply(
        lambda row: (
            f"{clean_text(row['stability_category']).replace('_', ' ')}; "
            f"stability index {row['stability_index']:.1f}; "
            f"{int(row['n_change_points'])} detected change points "
            f"across {int(row['transition_count'])} consecutive transitions"
        ),
        axis=1,
    )

    return result.drop_duplicates("company")


def temporal_profile(row: pd.Series) -> str:
    """Return a restrained temporal qualification from frozen WP5 evidence."""
    category = clean_text(row.get("stability_category", ""))
    mapping = {
        "most_stable": "most stable within the comparison",
        "relatively_stable": "relatively stable",
        "relatively_dynamic": "relatively dynamic",
        "most_dynamic": "most dynamic within the comparison",
    }
    if category in mapping:
        return mapping[category]

    index = row.get("stability_index", np.nan)
    if pd.notna(index):
        if index >= 75:
            return "relatively stable"
        if index <= 25:
            return "relatively dynamic"
    return "intermediate temporal stability"


def resolve_pairwise(frame: pd.DataFrame) -> pd.DataFrame:
    required = [
        "company_a",
        "company_b",
        "overall_differentiation_index",
        "differentiation_rank",
    ]
    resolved_columns = {}
    for candidate in required:
        resolved_columns[candidate] = first_existing_column(
            frame,
            [candidate],
            required=True,
            label=candidate,
        )

    result = pd.DataFrame(
        {
            "company_a": frame[resolved_columns["company_a"]].map(clean_text),
            "company_b": frame[resolved_columns["company_b"]].map(clean_text),
            "overall_differentiation_index": pd.to_numeric(
                frame[resolved_columns["overall_differentiation_index"]],
                errors="coerce",
            ),
            "differentiation_rank": pd.to_numeric(
                frame[resolved_columns["differentiation_rank"]],
                errors="coerce",
            ).astype("Int64"),
        }
    )

    for output_name, candidates in {
        "largest_dimension_contrast": [
            "largest_dimension_contrast",
        ],
        "largest_dimension_direction": [
            "largest_dimension_direction",
        ],
        "overall_difference_level": [
            "overall_difference_level",
        ],
        "pairwise_interpretation": [
            "pairwise_interpretation",
        ],
    }.items():
        result[output_name] = text_series(frame, candidates)

    return result


def nearest_and_furthest(
    company: str,
    pairwise: pd.DataFrame,
) -> dict[str, object]:
    subset = pairwise[
        (pairwise["company_a"] == company)
        | (pairwise["company_b"] == company)
    ].copy()

    if subset.empty:
        return {
            "closest_company": "",
            "closest_company_index": np.nan,
            "most_differentiated_company": "",
            "most_differentiated_company_index": np.nan,
            "most_differentiated_pair_rank": pd.NA,
            "principal_pairwise_contrast": "",
            "principal_pairwise_direction": "",
        }

    subset["other_company"] = np.where(
        subset["company_a"] == company,
        subset["company_b"],
        subset["company_a"],
    )

    closest = subset.sort_values(
        "overall_differentiation_index",
        ascending=True,
    ).iloc[0]
    furthest = subset.sort_values(
        "overall_differentiation_index",
        ascending=False,
    ).iloc[0]

    return {
        "closest_company": closest["other_company"],
        "closest_company_index": closest[
            "overall_differentiation_index"
        ],
        "most_differentiated_company": furthest["other_company"],
        "most_differentiated_company_index": furthest[
            "overall_differentiation_index"
        ],
        "most_differentiated_pair_rank": furthest[
            "differentiation_rank"
        ],
        "principal_pairwise_contrast": furthest[
            "largest_dimension_contrast"
        ],
        "principal_pairwise_direction": furthest[
            "largest_dimension_direction"
        ],
    }


def resolve_community_label_columns(
    dictionary: pd.DataFrame,
) -> tuple[str | None, str | None]:
    community_column = first_existing_column(
        dictionary,
        ["community", "community_id", "cluster"],
    )
    label_column = first_existing_column(
        dictionary,
        [
            "semantic_label",
            "community_semantic_label",
            "manual_semantic_label",
            "semantic_description",
            "community_description",
            "interpretation",
            "representative_evidence",
        ],
    )
    return community_column, label_column



def format_community_name(value: object) -> str:
    """
    Standardise community identifiers for publication-ready display.

    Examples
    --------
    13 -> Community 13
    "13" -> Community 13
    "Community13" -> Community 13
    "Community 13" -> Community 13
    """
    raw = clean_text(value)
    match = re.fullmatch(
        r"(?:community\s*)?(\d+)",
        raw,
        flags=re.IGNORECASE,
    )
    if match:
        return f"Community {match.group(1)}"
    return raw


def indefinite_article(phrase: str) -> str:
    """
    Return a grammatically suitable indefinite article for the identity label.
    """
    first = clean_text(phrase).lower()
    if first.startswith(("a", "e", "i", "o", "u")):
        return "an"
    return "a"


def resolve_top_communities(
    rankings: pd.DataFrame,
    dictionary: pd.DataFrame,
    top_n: int = 3,
) -> pd.DataFrame:
    rankings = resolve_company(rankings, "community_rankings.csv")

    community_column = first_existing_column(
        rankings,
        ["community", "community_id", "cluster"],
        required=True,
        label="community identifier in community_rankings.csv",
    )
    rank_column = first_existing_column(
        rankings,
        [
            "contribution_rank",
            "community_rank",
            "rank",
            "company_rank",
        ],
    )
    score_column = first_existing_column(
        rankings,
        [
            "contribution_score",
            "community_contribution",
            "weighted_contribution",
        ],
    )

    working = rankings.copy()
    working["community_key"] = working[community_column].map(clean_text)

    if rank_column is not None:
        working["_rank"] = pd.to_numeric(
            working[rank_column],
            errors="coerce",
        )
        working = working.sort_values(
            ["company", "_rank"],
            ascending=[True, True],
        )
    elif score_column is not None:
        working["_score"] = pd.to_numeric(
            working[score_column],
            errors="coerce",
        )
        working = working.sort_values(
            ["company", "_score"],
            ascending=[True, False],
        )
    else:
        working = working.sort_values(
            ["company", "community_key"]
        )

    dictionary_community, dictionary_label = (
        resolve_community_label_columns(dictionary)
    )

    if (
        dictionary_community is not None
        and dictionary_label is not None
    ):
        mapping = (
            dictionary[
                [dictionary_community, dictionary_label]
            ]
            .drop_duplicates(dictionary_community)
            .assign(
                community_key=lambda x: x[
                    dictionary_community
                ].map(clean_text),
                community_label=lambda x: x[
                    dictionary_label
                ].map(clean_text),
            )
            [["community_key", "community_label"]]
        )
        working = working.merge(
            mapping,
            on="community_key",
            how="left",
        )
    else:
        working["community_label"] = ""

    working["community_name"] = working[
        "community_key"
    ].map(format_community_name)

    working["community_display"] = np.where(
        working["community_label"].ne(""),
        working["community_name"]
        + " — "
        + working["community_label"],
        working["community_name"],
    )

    top = working.groupby(
        "company",
        observed=True,
        group_keys=False,
    ).head(top_n)

    return (
        top.groupby("company", observed=True)["community_display"]
        .apply(lambda values: "; ".join(values))
        .reset_index(name="dominant_communities")
    )


def identity_label(archetype: str) -> str:
    """Map internal archetypes to restrained publication-facing labels."""
    mapping = {
        "Adaptive Explorer": "exploration-led semantic profile",
        "Dynamic Generalist": "broad and adaptive semantic profile",
        "Stable Integrator": "integration-led stable semantic profile",
        "Balanced Curator": "balanced semantic profile",
    }
    return mapping.get(archetype, "organisational semantic profile")


def distinctiveness_label(rank: object, total: int) -> str:
    if pd.isna(rank):
        return "not ranked"
    rank = int(rank)
    if rank == 1:
        return "highest relative distinctiveness"
    if rank == total:
        return "lowest relative distinctiveness"
    if rank == 2:
        return "high relative distinctiveness"
    return "moderate relative distinctiveness"


def archetype_dimension_alignment(row: pd.Series) -> bool:
    archetype = clean_text(row["semantic_archetype"])
    highest = max(
        {
            "exploration": row["exploration_score"],
            "integration": row["integration_score"],
            "breadth": row["breadth_score"],
            "adaptation": row["adaptation_score"],
        },
        key=lambda key: {
            "exploration": row["exploration_score"],
            "integration": row["integration_score"],
            "breadth": row["breadth_score"],
            "adaptation": row["adaptation_score"],
        }[key],
    )

    expected = {
        "Adaptive Explorer": {"exploration"},
        "Dynamic Generalist": {"breadth", "adaptation"},
        "Stable Integrator": {"integration"},
        "Balanced Curator": {
            "exploration",
            "integration",
            "breadth",
            "adaptation",
        },
    }
    return highest in expected.get(archetype, {highest})


def temporal_archetype_alignment(row: pd.Series) -> bool:
    archetype = clean_text(row["semantic_archetype"])
    profile = clean_text(row["temporal_profile"])

    expected = {
        "Adaptive Explorer": {
            "selectively adaptive",
            "dynamic",
            "intermediate",
        },
        "Dynamic Generalist": {
            "highly dynamic",
            "dynamic",
        },
        "Stable Integrator": {
            "comparatively stable",
            "stable",
        },
        "Balanced Curator": {
            "moderately stable",
            "stable",
            "intermediate",
        },
    }
    return profile in expected.get(archetype, {profile})


def evidence_strength(
    distinctiveness_available: bool,
    communities_available: bool,
    temporal_available: bool,
    archetype_alignment: bool,
    temporal_alignment: bool,
    pairwise_available: bool,
) -> tuple[str, int]:
    checks = [
        distinctiveness_available,
        communities_available,
        temporal_available,
        archetype_alignment,
        temporal_alignment,
        pairwise_available,
    ]
    count = int(sum(bool(value) for value in checks))

    if count == 6:
        return "very strong", count
    if count >= 5:
        return "strong", count
    if count >= 4:
        return "moderate", count
    return "limited", count


def consistency_label(
    archetype_alignment: bool,
    temporal_alignment: bool,
    evidence_count: int,
) -> str:
    aligned = int(archetype_alignment) + int(temporal_alignment)

    if aligned == 2 and evidence_count >= 5:
        return "very high"
    if aligned >= 1 and evidence_count >= 4:
        return "high"
    if evidence_count >= 3:
        return "moderate"
    return "limited"


def narrative(row: pd.Series, company_count: int) -> str:
    """Build a restrained, evidence-traceable publication narrative."""
    company = row["company"]
    identity = row["integrated_semantic_identity"]
    distinctiveness = row["distinctiveness_label"]
    rank = row["signature_rank"]
    strength = row["signature_strength"]
    dominant = clean_text(row["positive_dominant_dimensions"])
    communities = clean_text(row["dominant_communities"])
    temporal = row["temporal_profile"]
    stability_index = row.get("stability_index", np.nan)
    closest = row["closest_company"]
    furthest = row["most_differentiated_company"]
    contrast = row["principal_pairwise_contrast"]

    rank_text = (
        f"ranked {int(rank)} of {company_count}"
        if pd.notna(rank)
        else "not assigned a distinctiveness rank"
    )
    strength_text = (
        f", with signature strength {strength:.3f}"
        if pd.notna(strength)
        else ""
    )

    if dominant and dominant != "none":
        dominant_phrase = dominant.replace(";", " and")
        fingerprint_sentence = (
            f"Its strongest positive fingerprint dimensions are "
            f"{dominant_phrase}."
        )
    else:
        fingerprint_sentence = (
            "Its fingerprint contains no single strongly dominant "
            "positive dimension."
        )

    temporal_sentence = f"Temporal evidence classifies the profile as {temporal}"
    if pd.notna(stability_index):
        temporal_sentence += f" (relative stability index {stability_index:.1f})"
    temporal_sentence += "."

    paragraph = (
        f"{company} exhibits an {identity} with {distinctiveness} "
        f"({rank_text}{strength_text}). {fingerprint_sentence} "
        f"Its leading semantic-community contributions are "
        f"{communities or 'not available'}. {temporal_sentence} "
        f"Within the four-company comparison, {company} is closest to "
        f"{closest or 'an unresolved comparator'} and most differentiated "
        f"from {furthest or 'an unresolved comparator'}"
    )
    if contrast:
        paragraph += f", with the latter contrast driven primarily by {contrast}"
    paragraph += (
        ". This interpretation summarises quantitative evidence within the "
        "analysed lyrical corpus and does not imply causal corporate culture "
        "or organisational intent."
    )
    return paragraph


def main() -> None:
    section("WP8f : INTEGRATED ORGANISATIONAL SEMANTIC IDENTITY")
    print(f"Script version          : {SCRIPT_VERSION}")
    print(f"Project root            : {ROOT}")
    print("Statistical inference   : none")
    print("New composite metrics   : none")

    for path in [
        SIGNATURE_SUMMARY_PATH,
        SIGNATURE_RANKINGS_PATH,
        COMMUNITY_RANKINGS_PATH,
        COMMUNITY_DICTIONARY_PATH,
        TRANSITION_SUMMARY_PATH,
        TEMPORAL_STABILITY_PATH,
        ARCHETYPES_PATH,
        DIFFERENTIATION_PATH,
    ]:
        require(path)

    section("LOADING WP8 EVIDENCE")

    signature_summary = load_csv(SIGNATURE_SUMMARY_PATH)
    signature_rankings = load_csv(SIGNATURE_RANKINGS_PATH)
    community_rankings = load_csv(COMMUNITY_RANKINGS_PATH)
    community_dictionary = load_csv(COMMUNITY_DICTIONARY_PATH)
    transitions = load_csv(TRANSITION_SUMMARY_PATH)
    temporal_stability = load_csv(TEMPORAL_STABILITY_PATH)
    archetypes = load_csv(ARCHETYPES_PATH)
    differentiation = load_csv(DIFFERENTIATION_PATH)

    signatures = resolve_signature(
        signature_summary,
        signature_rankings,
    )
    archetype_table = resolve_archetypes(archetypes)
    # WP8c remains available for transition-level community explanations,
    # but frozen WP5 evidence is the source of temporal magnitudes and
    # stability qualifications used in the integrated identity.
    transition_table = resolve_stability(temporal_stability)
    pairwise_table = resolve_pairwise(differentiation)
    community_table = resolve_top_communities(
        community_rankings,
        community_dictionary,
        top_n=3,
    )

    companies = sorted(archetype_table["company"].unique())
    company_count = len(companies)

    section("INTEGRATING ORGANISATIONAL IDENTITIES")

    integrated = archetype_table.merge(
        signatures,
        on="company",
        how="left",
        validate="one_to_one",
    )
    integrated = integrated.merge(
        transition_table,
        on="company",
        how="left",
        validate="one_to_one",
    )
    integrated = integrated.merge(
        community_table,
        on="company",
        how="left",
        validate="one_to_one",
    )

    pairwise_records = [
        {
            "company": company,
            **nearest_and_furthest(company, pairwise_table),
        }
        for company in companies
    ]
    integrated = integrated.merge(
        pd.DataFrame(pairwise_records),
        on="company",
        how="left",
        validate="one_to_one",
    )

    integrated["temporal_profile"] = integrated.apply(
        temporal_profile,
        axis=1,
    )
    integrated["integrated_semantic_identity"] = integrated[
        "semantic_archetype"
    ].map(identity_label)
    integrated["distinctiveness_label"] = integrated[
        "signature_rank"
    ].map(
        lambda rank: distinctiveness_label(
            rank,
            company_count,
        )
    )

    integrated["archetype_dimension_alignment"] = integrated.apply(
        archetype_dimension_alignment,
        axis=1,
    )
    integrated["temporal_archetype_alignment"] = integrated.apply(
        temporal_archetype_alignment,
        axis=1,
    )

    strength_results = integrated.apply(
        lambda row: evidence_strength(
            distinctiveness_available=pd.notna(
                row["signature_strength"]
            ),
            communities_available=bool(
                clean_text(row["dominant_communities"])
            ),
            temporal_available=(
                pd.notna(row["mean_transition_js"])
                or pd.notna(row["mean_transition_cosine"])
                or bool(clean_text(row["temporal_interpretation"]))
            ),
            archetype_alignment=bool(
                row["archetype_dimension_alignment"]
            ),
            temporal_alignment=bool(
                row["temporal_archetype_alignment"]
            ),
            pairwise_available=bool(
                clean_text(row["most_differentiated_company"])
            ),
        ),
        axis=1,
    )

    integrated["evidence_strength"] = [
        value[0] for value in strength_results
    ]
    integrated["evidence_support_count"] = [
        value[1] for value in strength_results
    ]
    integrated["overall_consistency"] = integrated.apply(
        lambda row: consistency_label(
            bool(row["archetype_dimension_alignment"]),
            bool(row["temporal_archetype_alignment"]),
            int(row["evidence_support_count"]),
        ),
        axis=1,
    )

    integrated["identity_summary"] = integrated.apply(
        lambda row: (
            f"{row['company']} exhibits "
            f"{indefinite_article(row['integrated_semantic_identity'])} "
            f"{row['integrated_semantic_identity']} with "
            f"{row['distinctiveness_label']}, a "
            f"{row['temporal_profile']} temporal qualification, and greatest "
            f"relative differentiation from "
            f"{row['most_differentiated_company']}."
        ),
        axis=1,
    )

    integrated["publication_narrative"] = integrated.apply(
        lambda row: clean_text(narrative(row, company_count)),
        axis=1,
    )

    section("BUILDING TRACEABILITY TABLE")

    traceability_records: list[dict[str, object]] = []

    for row in integrated.itertuples(index=False):
        claims = [
            (
                "relative organisational distinctiveness",
                (
                    f"{row.distinctiveness_label}; "
                    f"rank {row.signature_rank} of {company_count}; "
                    f"signature strength {row.signature_strength:.3f}"
                    if pd.notna(row.signature_strength)
                    else row.distinctiveness_label
                ),
                "WP8a",
                "company_signature_summary.csv; "
                "company_signature_rankings.csv",
            ),
            (
                "dominant semantic communities",
                row.dominant_communities,
                "WP8b",
                "community_rankings.csv; "
                "community_semantic_dictionary.csv",
            ),
            (
                "temporal semantic behaviour",
                (
                    f"{row.temporal_profile}; "
                    f"mean JS {row.mean_transition_js:.5f}; "
                    f"mean cosine {row.mean_transition_cosine:.5f}"
                    if pd.notna(row.mean_transition_js)
                    and pd.notna(row.mean_transition_cosine)
                    else row.temporal_profile
                ),
                "WP8c",
                "company_transition_summary.csv",
            ),
            (
                "organisational semantic archetype",
                (
                    f"{row.semantic_archetype}; "
                    f"fingerprint {row.semantic_fingerprint}"
                ),
                "WP8d",
                "organisational_semantic_archetypes.csv",
            ),
            (
                "relative pairwise differentiation",
                (
                    f"closest to {row.closest_company}; "
                    f"most differentiated from "
                    f"{row.most_differentiated_company}; "
                    f"principal contrast "
                    f"{row.principal_pairwise_contrast}"
                ),
                "WP8e",
                "company_differentiation_rankings.csv",
            ),
            (
                "integrated organisational semantic identity",
                row.integrated_semantic_identity,
                "WP8f synthesis",
                "Derived directly from WP8a-WP8e; "
                "no new statistical inference",
            ),
        ]

        for claim_type, claim, work_package, source_file in claims:
            traceability_records.append(
                {
                    "company": row.company,
                    "claim_type": claim_type,
                    "claim": clean_text(claim),
                    "supporting_work_package": work_package,
                    "source_file": source_file,
                }
            )

    traceability = pd.DataFrame(traceability_records)

    section("BUILDING CONSISTENCY TABLE")

    consistency = integrated[
        [
            "company",
            "semantic_archetype",
            "integrated_semantic_identity",
            "archetype_dimension_alignment",
            "temporal_archetype_alignment",
            "signature_rank",
            "dominant_communities",
            "closest_company",
            "most_differentiated_company",
            "evidence_support_count",
            "evidence_strength",
            "overall_consistency",
        ]
    ].copy()

    consistency["consistency_interpretation"] = consistency.apply(
        lambda row: (
            f"{row['company']} shows "
            f"{row['overall_consistency']} consistency across WP8 analyses. "
            f"The latent dimensions "
            f"{'support' if row['archetype_dimension_alignment'] else 'do not fully support'} "
            f"the assigned archetype, and temporal behaviour "
            f"{'supports' if row['temporal_archetype_alignment'] else 'does not fully support'} "
            f"the archetype-level interpretation."
        ),
        axis=1,
    )

    summary = integrated[
        [
            "company",
            "signature_rank",
            "distinctiveness_label",
            "semantic_archetype",
            "integrated_semantic_identity",
            "dominant_communities",
            "temporal_profile",
            "closest_company",
            "most_differentiated_company",
            "principal_pairwise_contrast",
            "evidence_strength",
            "overall_consistency",
            "identity_summary",
        ]
    ].copy()

    narratives = integrated[
        [
            "company",
            "semantic_archetype",
            "integrated_semantic_identity",
            "publication_narrative",
        ]
    ].copy()

    integrated_columns = [
        "company",
        "signature_strength",
        "signature_rank",
        "distinctiveness_label",
        "semantic_archetype",
        "integrated_semantic_identity",
        "semantic_fingerprint",
        "exploration_score",
        "integration_score",
        "breadth_score",
        "adaptation_score",
        "dominant_dimension",
        "positive_dominant_dimensions",
        "dominance_type",
        "dominant_communities",
        "transition_count",
        "mean_transition_js",
        "max_transition_js",
        "mean_transition_cosine",
        "median_transition_js",
        "std_transition_js",
        "n_change_points",
        "change_point_rate",
        "stability_index",
        "stability_rank",
        "stability_category",
        "largest_change_transition",
        "largest_change_js",
        "largest_change_severity",
        "temporal_profile",
        "temporal_interpretation",
        "closest_company",
        "closest_company_index",
        "most_differentiated_company",
        "most_differentiated_company_index",
        "most_differentiated_pair_rank",
        "principal_pairwise_contrast",
        "principal_pairwise_direction",
        "archetype_dimension_alignment",
        "temporal_archetype_alignment",
        "evidence_support_count",
        "evidence_strength",
        "overall_consistency",
        "identity_summary",
        "publication_narrative",
    ]

    integrated = integrated[integrated_columns].sort_values(
        "signature_rank",
        na_position="last",
    )

    integrated.to_csv(
        INTEGRATED_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    narratives.to_csv(
        NARRATIVES_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    narrative_blocks = []
    for row in narratives.itertuples(index=False):
        narrative_blocks.append(
            f"{row.company} — {row.integrated_semantic_identity}\n"
            f"{clean_text(row.publication_narrative)}"
        )
    NARRATIVES_TEXT_PATH.write_text(
        "\n\n".join(narrative_blocks) + "\n",
        encoding="utf-8",
    )
    traceability.to_csv(
        TRACEABILITY_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    consistency.to_csv(
        CONSISTENCY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "script": Path(__file__).name,
        "script_version": SCRIPT_VERSION,
        "analysis_name": (
            "WP8f Integrated Organisational Semantic Identity"
        ),
        "methodological_status": {
            "new_statistical_inference": False,
            "new_hypothesis_tests": False,
            "new_latent_variables": False,
            "new_composite_metrics": False,
            "description": (
                "This script synthesises previously computed WP8a-WP8e "
                "results. It does not alter, recompute, or statistically "
                "extend the underlying analyses."
            ),
        },
        "companies": companies,
        "input_files": {
            "WP8a_signature_summary": str(SIGNATURE_SUMMARY_PATH),
            "WP8a_signature_rankings": str(SIGNATURE_RANKINGS_PATH),
            "WP8b_community_rankings": str(COMMUNITY_RANKINGS_PATH),
            "WP8b_community_dictionary": str(COMMUNITY_DICTIONARY_PATH),
            "WP8c_transition_summary": str(TRANSITION_SUMMARY_PATH),
            "WP5_temporal_stability": str(TEMPORAL_STABILITY_PATH),
            "WP8d_archetypes": str(ARCHETYPES_PATH),
            "WP8e_differentiation": str(DIFFERENTIATION_PATH),
        },
        "integrated_identity_mapping": {
            "Adaptive Explorer": (
                "innovation-oriented semantic specialist"
            ),
            "Dynamic Generalist": (
                "adaptive semantic generalist"
            ),
            "Stable Integrator": (
                "integrated semantic ecosystem"
            ),
            "Balanced Curator": (
                "curated semantic portfolio"
            ),
        },
        "evidence_strength": {
            "description": (
                "Qualitative completeness and agreement label, not "
                "statistical confidence."
            ),
            "very strong": "All six evidence checks are supported.",
            "strong": "Five of six evidence checks are supported.",
            "moderate": "Four of six evidence checks are supported.",
            "limited": "Fewer than four evidence checks are supported.",
        },
        "consistency": {
            "description": (
                "Deterministic qualitative assessment of whether latent "
                "dimensions and temporal behaviour align with the assigned "
                "archetype, conditional on evidence completeness."
            )
        },
        "outputs": {
            "integrated_company_identity": str(INTEGRATED_PATH),
            "company_identity_summary": str(SUMMARY_PATH),
            "company_identity_narratives": str(NARRATIVES_PATH),
            "company_identity_narratives_text": str(NARRATIVES_TEXT_PATH),
            "identity_evidence_traceability": str(TRACEABILITY_PATH),
            "cross_analysis_consistency": str(CONSISTENCY_PATH),
        },
    }

    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    section("INTEGRATED ORGANISATIONAL IDENTITIES")

    display_columns = [
        "company",
        "signature_rank",
        "semantic_archetype",
        "integrated_semantic_identity",
        "temporal_profile",
        "closest_company",
        "most_differentiated_company",
        "evidence_strength",
        "overall_consistency",
    ]

    print(
        integrated[display_columns].to_string(
            index=False,
        )
    )

    section("PUBLICATION-READY NARRATIVES")

    for row in narratives.itertuples(index=False):
        print(f"\n{row.company} — {row.integrated_semantic_identity}")
        sentences = re.split(r"(?<=[.!?])\s+", clean_text(row.publication_narrative))
        for sentence in sentences:
            if sentence:
                print(sentence)

    section("OUTPUTS")
    print(f"Integrated identities   : {INTEGRATED_PATH}")
    print(f"Identity summary        : {SUMMARY_PATH}")
    print(f"Narratives (CSV)        : {NARRATIVES_PATH}")
    print(f"Narratives (text)       : {NARRATIVES_TEXT_PATH}")
    print(f"Evidence traceability   : {TRACEABILITY_PATH}")
    print(f"Consistency assessment  : {CONSISTENCY_PATH}")
    print(f"Metadata                : {METADATA_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"\nWP8f failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        raise
