#!/usr/bin/env python3
"""
WP8d: Organisational Semantic Archetypes

Purpose
-------
Infer reproducible organisational semantic archetypes from four higher-order
dimensions derived from the frozen WP8a-WP8c outputs:

1. Exploration
2. Structural integration
3. Semantic breadth
4. Temporal adaptation

The dimensions are constructed from standardised company-level metrics and reported with ranks, dominance categories, and a compact semantic fingerprint.
Archetype labels are assigned deterministically using a dominance-plus-evidence rule.

Important interpretation
------------------------
The scores are relative to the four-company study. They are comparative,
not absolute industry-wide measurements.

Inputs
------
outputs/explainability/organisational_semantic_profiles.csv
outputs/explainability/organisational_profile_evidence.csv

Outputs
-------
outputs/explainability/organisational_semantic_archetypes.csv
outputs/explainability/organisational_archetype_dimensions.csv
outputs/explainability/organisational_archetype_evidence.csv
outputs/explainability/organisational_archetype_metadata.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_VERSION = "2026-07-19-v3"

HIGH_THRESHOLD = 0.50
LOW_THRESHOLD = -0.50


def find_project_root(start: Path) -> Path:
    for candidate in [start.resolve(), Path.cwd().resolve()]:
        for parent in [candidate, *candidate.parents]:
            if (parent / "outputs" / "tables").exists():
                return parent
    raise FileNotFoundError(
        "Could not locate the project root containing outputs/tables."
    )


ROOT = find_project_root(Path(__file__).parent)
OUT = ROOT / "outputs" / "explainability"

PROFILES_PATH = OUT / "organisational_semantic_profiles.csv"
PROFILE_EVIDENCE_PATH = OUT / "organisational_profile_evidence.csv"

ARCHETYPES_PATH = OUT / "organisational_semantic_archetypes.csv"
DIMENSIONS_PATH = OUT / "organisational_archetype_dimensions.csv"
EVIDENCE_PATH = OUT / "organisational_archetype_evidence.csv"
METADATA_PATH = OUT / "organisational_archetype_metadata.json"


def section(title: str) -> None:
    line = "=" * 78
    print(f"\n{line}\n{title}\n{line}")


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    text = "".join(
        char
        for char in text
        if not (ord(char) < 32 or 127 <= ord(char) <= 159)
    )
    return " ".join(
        text.replace("\u00a0", " ")
        .replace("\u2007", " ")
        .replace("\u202f", " ")
        .replace("\u200b", "")
        .replace("\ufeff", "")
        .split()
    )


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    mean = values.mean()
    std = values.std(ddof=0)

    if not np.isfinite(std) or std <= 0:
        return pd.Series(0.0, index=series.index)

    return (values - mean) / std


def row_mean(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    return frame[columns].mean(axis=1, skipna=True)


def classify_level(score: float) -> str:
    if score >= HIGH_THRESHOLD:
        return "high"
    if score <= LOW_THRESHOLD:
        return "low"
    return "moderate"


def breadth_descriptor(score: float) -> str:
    if score >= HIGH_THRESHOLD:
        return "broad"
    if score <= LOW_THRESHOLD:
        return "focused"
    return "balanced"


def temporal_descriptor(score: float) -> str:
    if score >= HIGH_THRESHOLD:
        return "adaptive"
    if score <= LOW_THRESHOLD:
        return "stable"
    return "evolving"


def exploration_descriptor(score: float) -> str:
    if score >= HIGH_THRESHOLD:
        return "exploratory"
    if score <= LOW_THRESHOLD:
        return "conservative"
    return "selective"


def integration_descriptor(score: float) -> str:
    if score >= HIGH_THRESHOLD:
        return "integrated"
    if score <= LOW_THRESHOLD:
        return "fragmented"
    return "connected"


def assign_archetype(
    exploration: float,
    integration: float,
    breadth: float,
    adaptation: float,
) -> str:
    """
    Assign a deterministic archetype using dominance plus supporting evidence.

    A dimension is considered positively dominant only when its score is at
    least HIGH_THRESHOLD. Scores below that threshold are not treated as
    substantively dominant merely because they are the largest among four
    near-zero values.
    """
    high = {
        "exploration": exploration >= HIGH_THRESHOLD,
        "integration": integration >= HIGH_THRESHOLD,
        "breadth": breadth >= HIGH_THRESHOLD,
        "adaptation": adaptation >= HIGH_THRESHOLD,
    }
    low = {
        "exploration": exploration <= LOW_THRESHOLD,
        "integration": integration <= LOW_THRESHOLD,
        "breadth": breadth <= LOW_THRESHOLD,
        "adaptation": adaptation <= LOW_THRESHOLD,
    }

    if high["exploration"] and low["breadth"] and not low["adaptation"]:
        return "Adaptive Explorer"

    if high["exploration"] and high["adaptation"]:
        return "Adaptive Explorer"

    if high["integration"] and low["adaptation"]:
        return "Stable Integrator"

    if high["breadth"] and high["adaptation"]:
        return "Dynamic Generalist"

    if high["breadth"] and high["integration"]:
        return "Integrated Generalist"

    if low["breadth"] and low["exploration"] and low["adaptation"]:
        return "Stable Specialist"

    if high["exploration"]:
        return "Selective Explorer"

    if high["integration"]:
        return "Semantic Integrator"

    if high["breadth"]:
        return "Broad Generalist"

    if high["adaptation"]:
        return "Adaptive Repositioner"

    if low["exploration"] and not high["breadth"]:
        return "Balanced Curator"

    if low["adaptation"] and not high["exploration"]:
        return "Stable Curator"

    return "Balanced Portfolio"


def evidence_statement(
    company: str,
    archetype: str,
    row: pd.Series,
) -> str:
    exploration = row["exploration_score"]
    integration = row["integration_score"]
    breadth = row["breadth_score"]
    adaptation = row["adaptation_score"]
    dominance_type = row["dominance_type"]
    dominant_dimensions = row["positive_dominant_dimensions"]

    if dominance_type == "multiple":
        dominance_clause = (
            f"It exhibits multiple dominant dimensions "
            f"({dominant_dimensions.replace(';', ' and')}). "
        )
    elif dominance_type == "single":
        dominance_clause = (
            f"It exhibits one dominant dimension "
            f"({dominant_dimensions}). "
        )
    else:
        dominance_clause = (
            "No latent dimension exceeds the positive dominance threshold. "
        )

    return (
        f"{company} is classified as a {archetype}. "
        f"{dominance_clause}"
        f"Its organisational profile is "
        f"{exploration_descriptor(exploration)} in exploration, "
        f"{integration_descriptor(integration)} in structural integration, "
        f"{breadth_descriptor(breadth)} in semantic breadth, and "
        f"{temporal_descriptor(adaptation)} in temporal behaviour. "
        f"The relative dimension scores are "
        f"{exploration:.3f} for exploration, "
        f"{integration:.3f} for integration, "
        f"{breadth:.3f} for breadth, and "
        f"{adaptation:.3f} for adaptation."
    )


def strategic_interpretation(
    company: str,
    archetype: str,
    row: pd.Series,
) -> str:
    if archetype == "Adaptive Explorer":
        return (
            f"{company} combines strong movement toward novel semantic regions "
            f"with a focused catalogue and a non-stable temporal profile, "
            f"indicating an organisational identity built through selective "
            f"and continuing experimentation."
        )

    if archetype == "Stable Integrator":
        return (
            f"{company} maintains a highly connected semantic ecosystem while "
            f"changing comparatively slowly over time, indicating a mature and "
            f"cohesive organisational identity."
        )

    if archetype == "Dynamic Generalist":
        return (
            f"{company} spans a broad semantic space while repeatedly "
            f"reconfiguring its internal emphasis, indicating continual "
            f"repositioning rather than optimisation around a fixed core."
        )

    if archetype == "Focused Curator":
        return (
            f"{company} concentrates on a comparatively selective semantic "
            f"portfolio with limited expansion into new regions, indicating "
            f"curation around a recognisable but less expansive identity."
        )

    if archetype == "Integrated Generalist":
        return (
            f"{company} combines broad semantic coverage with strong internal "
            f"connectivity, indicating a diversified yet coherent catalogue."
        )

    if archetype == "Stable Specialist":
        return (
            f"{company} maintains a narrow semantic portfolio that changes "
            f"gradually, indicating sustained exploitation of an established "
            f"organisational niche."
        )

    if archetype in {"Selective Explorer", "Adaptive Repositioner"}:
        return (
            f"{company} exhibits targeted semantic change rather than broad "
            f"catalogue expansion, indicating selective experimentation."
        )

    if archetype in {"Balanced Curator", "Stable Curator"}:
        return (
            f"{company} shows no strongly dominant positive organisational "
            f"dimension. Its comparatively low exploration and moderate "
            f"breadth indicate a curated, balanced semantic portfolio rather "
            f"than aggressive expansion or continual repositioning."
        )

    if archetype == "Balanced Portfolio":
        return (
            f"{company} shows no substantively dominant semantic dimension, "
            f"indicating a comparatively balanced organisational portfolio."
        )

    if archetype in {"Semantic Integrator", "Connected Integrator"}:
        return (
            f"{company} is primarily distinguished by the structural "
            f"connectivity of its semantic catalogue rather than extreme "
            f"breadth or temporal change."
        )

    return (
        f"{company} occupies an intermediate organisational position, with no "
        f"single semantic strategy overwhelmingly dominating the others."
    )


def main() -> None:
    section("WP8d : ORGANISATIONAL SEMANTIC ARCHETYPES")
    print(f"Script version          : {SCRIPT_VERSION}")
    print(f"Project root            : {ROOT}")
    print(f"Profile input           : {PROFILES_PATH}")
    print(f"Evidence input          : {PROFILE_EVIDENCE_PATH}")
    print(f"High threshold          : {HIGH_THRESHOLD:+.2f}")
    print(f"Low threshold           : {LOW_THRESHOLD:+.2f}")

    require(PROFILES_PATH)
    require(PROFILE_EVIDENCE_PATH)

    profiles = pd.read_csv(PROFILES_PATH)
    evidence = pd.read_csv(PROFILE_EVIDENCE_PATH)

    profiles["company"] = profiles["company"].map(clean_text)
    evidence["company"] = evidence["company"].map(clean_text)

    required_metrics = [
        "mean_centroid_novelty",
        "mean_local_novelty",
        "relative_organisational_distinctiveness",
        "mean_bridge_affinity",
        "mean_participation_coefficient",
        "mean_max_centroid_cosine",
        "normalised_semantic_entropy",
        "semantic_concentration_hhi",
        "community_coverage",
        "mean_js_divergence",
        "mean_cosine_similarity",
    ]

    missing = [
        column for column in required_metrics
        if column not in profiles.columns
    ]
    if missing:
        raise KeyError(
            "Missing required profile metrics: "
            + ", ".join(missing)
        )

    section("STANDARDISING ORGANISATIONAL METRICS")

    metric_directions = {
        "mean_centroid_novelty": 1,
        "mean_local_novelty": 1,
        "relative_organisational_distinctiveness": 1,
        "mean_bridge_affinity": 1,
        "mean_participation_coefficient": 1,
        "mean_max_centroid_cosine": 1,
        "normalised_semantic_entropy": 1,
        "semantic_concentration_hhi": -1,
        "community_coverage": 1,
        "mean_js_divergence": 1,
        "mean_cosine_similarity": -1,
    }

    z_columns = []

    for metric, direction in metric_directions.items():
        z_column = f"z_{metric}"
        profiles[z_column] = zscore(profiles[metric]) * direction
        z_columns.append(z_column)

    profiles["exploration_score"] = row_mean(
        profiles,
        [
            "z_mean_centroid_novelty",
            "z_mean_local_novelty",
            "z_relative_organisational_distinctiveness",
        ],
    )

    profiles["integration_score"] = row_mean(
        profiles,
        [
            "z_mean_bridge_affinity",
            "z_mean_participation_coefficient",
            "z_mean_max_centroid_cosine",
        ],
    )

    profiles["breadth_score"] = row_mean(
        profiles,
        [
            "z_normalised_semantic_entropy",
            "z_semantic_concentration_hhi",
            "z_community_coverage",
        ],
    )

    profiles["adaptation_score"] = row_mean(
        profiles,
        [
            "z_mean_js_divergence",
            "z_mean_cosine_similarity",
        ],
    )

    dimension_columns = [
        "exploration_score",
        "integration_score",
        "breadth_score",
        "adaptation_score",
    ]

    profiles["largest_dimension"] = profiles[
        dimension_columns
    ].idxmax(axis=1).str.replace("_score", "", regex=False)

    profiles["largest_dimension_score"] = profiles[
        dimension_columns
    ].max(axis=1)

    profiles["dominant_dimension"] = np.where(
        profiles["largest_dimension_score"] >= HIGH_THRESHOLD,
        profiles["largest_dimension"],
        "none",
    )

    profiles["exploration_level"] = profiles[
        "exploration_score"
    ].map(classify_level)
    profiles["integration_level"] = profiles[
        "integration_score"
    ].map(classify_level)
    profiles["breadth_level"] = profiles[
        "breadth_score"
    ].map(classify_level)
    profiles["adaptation_level"] = profiles[
        "adaptation_score"
    ].map(classify_level)

    profiles["positive_dominant_dimensions"] = profiles.apply(
        lambda row: "; ".join(
            dimension
            for dimension in [
                "exploration",
                "integration",
                "breadth",
                "adaptation",
            ]
            if row[f"{dimension}_score"] >= HIGH_THRESHOLD
        ),
        axis=1,
    )
    profiles["positive_dominant_dimensions"] = profiles[
        "positive_dominant_dimensions"
    ].replace("", "none")

    profiles["has_positive_dominance"] = profiles[
        "positive_dominant_dimensions"
    ].ne("none")

    profiles["dominance_count"] = profiles.apply(
        lambda row: sum(
            row[f"{dimension}_score"] >= HIGH_THRESHOLD
            for dimension in [
                "exploration",
                "integration",
                "breadth",
                "adaptation",
            ]
        ),
        axis=1,
    )

    profiles["dominance_type"] = profiles[
        "dominance_count"
    ].map(
        {
            0: "none",
            1: "single",
            2: "multiple",
            3: "multiple",
            4: "multiple",
        }
    )

    profiles["semantic_fingerprint"] = profiles.apply(
        lambda row: (
            f"({row['exploration_score']:+.2f}, "
            f"{row['integration_score']:+.2f}, "
            f"{row['breadth_score']:+.2f}, "
            f"{row['adaptation_score']:+.2f})"
        ),
        axis=1,
    )

    profiles["semantic_archetype"] = profiles.apply(
        lambda row: assign_archetype(
            row["exploration_score"],
            row["integration_score"],
            row["breadth_score"],
            row["adaptation_score"],
        ),
        axis=1,
    )

    profiles["archetype_evidence_statement"] = profiles.apply(
        lambda row: evidence_statement(
            row["company"],
            row["semantic_archetype"],
            row,
        ),
        axis=1,
    )

    profiles["strategic_interpretation"] = profiles.apply(
        lambda row: strategic_interpretation(
            row["company"],
            row["semantic_archetype"],
            row,
        ),
        axis=1,
    )

    for dimension in dimension_columns:
        rank_column = dimension.replace("_score", "_rank")
        profiles[rank_column] = (
            profiles[dimension]
            .rank(method="min", ascending=False)
            .astype("Int64")
        )

    section("BUILDING EVIDENCE TABLE")

    evidence_summary = (
        evidence.groupby(["company", "direction"], observed=True)
        .agg(
            communities=(
                "community",
                lambda values: "; ".join(
                    str(value) for value in values
                ),
            ),
            representative_songs=(
                "representative_songs",
                lambda values: " | ".join(
                    clean_text(value)
                    for value in values
                    if clean_text(value)
                ),
            ),
        )
        .reset_index()
    )

    positive = (
        evidence_summary.loc[
            evidence_summary["direction"].eq("positive")
        ]
        .drop(columns="direction")
        .rename(
            columns={
                "communities": "positive_communities",
                "representative_songs":
                    "positive_representative_songs",
            }
        )
    )

    negative = (
        evidence_summary.loc[
            evidence_summary["direction"].eq("negative")
        ]
        .drop(columns="direction")
        .rename(
            columns={
                "communities": "negative_communities",
                "representative_songs":
                    "negative_representative_songs",
            }
        )
    )

    archetype_evidence = (
        profiles[
            [
                "company",
                "semantic_archetype",
                "exploration_score",
                "integration_score",
                "breadth_score",
                "adaptation_score",
                "dominance_type",
                "positive_dominant_dimensions",
                "semantic_fingerprint",
                "archetype_evidence_statement",
                "strategic_interpretation",
            ]
        ]
        .merge(positive, on="company", how="left")
        .merge(negative, on="company", how="left")
    )

    archetype_columns = [
        "company",
        "semantic_archetype",
        "dominant_dimension",
        "largest_dimension",
        "largest_dimension_score",
        "positive_dominant_dimensions",
        "has_positive_dominance",
        "dominance_count",
        "dominance_type",
        "semantic_fingerprint",
        "exploration_score",
        "exploration_level",
        "exploration_rank",
        "integration_score",
        "integration_level",
        "integration_rank",
        "breadth_score",
        "breadth_level",
        "breadth_rank",
        "adaptation_score",
        "adaptation_level",
        "adaptation_rank",
        "relative_organisational_distinctiveness",
        "normalised_semantic_entropy",
        "semantic_concentration_hhi",
        "community_coverage",
        "mean_centroid_novelty",
        "mean_bridge_affinity",
        "mean_participation_coefficient",
        "mean_js_divergence",
        "mean_cosine_similarity",
        "archetype_evidence_statement",
        "strategic_interpretation",
    ]

    archetypes = profiles[archetype_columns].copy()

    dimension_records = []

    dimension_definitions = {
        "exploration": [
            "mean_centroid_novelty",
            "mean_local_novelty",
            "relative_organisational_distinctiveness",
        ],
        "integration": [
            "mean_bridge_affinity",
            "mean_participation_coefficient",
            "mean_max_centroid_cosine",
        ],
        "breadth": [
            "normalised_semantic_entropy",
            "inverse_semantic_concentration_hhi",
            "community_coverage",
        ],
        "adaptation": [
            "mean_js_divergence",
            "inverse_mean_cosine_similarity",
        ],
    }

    for row in profiles.itertuples(index=False):
        for dimension in [
            "exploration",
            "integration",
            "breadth",
            "adaptation",
        ]:
            score = getattr(row, f"{dimension}_score")
            dimension_records.append(
                {
                    "company": row.company,
                    "dimension": dimension,
                    "score": float(score),
                    "level": classify_level(score),
                    "rank": int(
                        getattr(row, f"{dimension}_rank")
                    ),
                    "constituent_metrics": "; ".join(
                        dimension_definitions[dimension]
                    ),
                }
            )

    dimensions = pd.DataFrame(dimension_records)

    archetypes.to_csv(
        ARCHETYPES_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    dimensions.to_csv(
        DIMENSIONS_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    archetype_evidence.to_csv(
        EVIDENCE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "script": Path(__file__).name,
        "script_version": SCRIPT_VERSION,
        "analysis_name": "WP8d Organisational Semantic Archetypes",
        "scope": (
            "Comparative four-company organisational semantic analysis"
        ),
        "thresholds": {
            "high": HIGH_THRESHOLD,
            "low": LOW_THRESHOLD,
        },
        "dominance_rule": (
            "A dimension is considered positively dominant only when its "
            "score is at least the high threshold. When no dimension reaches "
            "that threshold, the archetype is assigned from the overall score "
            "pattern rather than from the numerically largest near-zero score. "
            "When multiple dimensions exceed the threshold, the archetype "
            "represents their joint organisational behaviour."
        ),
        "dominance_categories": {
            "single": (
                "One latent dimension exceeds the dominance threshold."
            ),
            "multiple": (
                "More than one latent dimension exceeds the dominance "
                "threshold."
            ),
            "none": (
                "No latent dimension exceeds the dominance threshold; the "
                "archetype is assigned from the overall score configuration."
            ),
        },
        "semantic_fingerprint_order": [
            "exploration",
            "integration",
            "breadth",
            "adaptation",
        ],
        "dimension_construction": {
            "exploration": {
                "metrics": [
                    "mean_centroid_novelty",
                    "mean_local_novelty",
                    "relative_organisational_distinctiveness",
                ],
                "direction": "higher_is_more_exploratory",
            },
            "integration": {
                "metrics": [
                    "mean_bridge_affinity",
                    "mean_participation_coefficient",
                    "mean_max_centroid_cosine",
                ],
                "direction": "higher_is_more_integrated",
            },
            "breadth": {
                "metrics": [
                    "normalised_semantic_entropy",
                    "inverse_semantic_concentration_hhi",
                    "community_coverage",
                ],
                "direction": "higher_is_broader",
            },
            "adaptation": {
                "metrics": [
                    "mean_js_divergence",
                    "inverse_mean_cosine_similarity",
                ],
                "direction": "higher_is_more_adaptive",
            },
        },
        "standardisation": (
            "Population z-scores computed across the four companies."
        ),
        "aggregation": (
            "Unweighted arithmetic mean of standardised constituent "
            "metrics within each dimension."
        ),
        "interpretation_caution": (
            "Archetypes are relative, deterministic summaries of the "
            "four-company sample. They should not be interpreted as "
            "universal industry categories without external validation."
        ),
        "outputs": {
            "archetypes": str(ARCHETYPES_PATH),
            "dimensions": str(DIMENSIONS_PATH),
            "evidence": str(EVIDENCE_PATH),
        },
    }

    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    section("ORGANISATIONAL SEMANTIC ARCHETYPES")

    display_columns = [
        "company",
        "semantic_archetype",
        "exploration_score",
        "exploration_rank",
        "integration_score",
        "integration_rank",
        "breadth_score",
        "breadth_rank",
        "adaptation_score",
        "adaptation_rank",
        "dominance_type",
        "dominant_dimension",
        "positive_dominant_dimensions",
        "semantic_fingerprint",
    ]

    print(
        archetypes[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    section("ARCHETYPE INTERPRETATIONS")

    for row in archetypes.itertuples(index=False):
        print(f"\n{row.company} — {row.semantic_archetype}")
        print(row.strategic_interpretation)

    section("OUTPUTS")
    print(f"Archetypes              : {ARCHETYPES_PATH}")
    print(f"Dimensions              : {DIMENSIONS_PATH}")
    print(f"Evidence                : {EVIDENCE_PATH}")
    print(f"Metadata                : {METADATA_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"\nWP8d failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        raise
