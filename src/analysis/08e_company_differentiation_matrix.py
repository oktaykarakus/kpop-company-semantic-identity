#!/usr/bin/env python3
"""
WP8e: Company Differentiation Matrix

Purpose
-------
Quantify how strongly each pair of companies differs across three complementary
levels of organisational semantics:

1. Latent organisational dimensions
   - exploration
   - integration
   - breadth
   - adaptation

2. Semantic-community composition
   - Jensen-Shannon divergence
   - total variation distance
   - cosine similarity

3. Organisational archetype structure
   - archetype agreement
   - dominance-pattern agreement
   - largest dimension contrasts

The script produces publication-ready long-form and square-matrix outputs,
plus a compact pairwise interpretation table. The composite index gives equal weight to latent-dimension distance and community Jensen-Shannon divergence.

Inputs
------
outputs/explainability/organisational_semantic_archetypes.csv
outputs/explainability/community_contributions.csv

Outputs
-------
outputs/explainability/company_differentiation_pairs.csv
outputs/explainability/company_differentiation_matrix.csv
outputs/explainability/company_dimension_distance_matrix.csv
outputs/explainability/company_community_js_matrix.csv
outputs/explainability/company_community_tv_matrix.csv
outputs/explainability/company_community_cosine_matrix.csv
outputs/explainability/company_differentiation_rankings.csv
outputs/explainability/company_differentiation_metadata.json
"""

from __future__ import annotations

import json
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_VERSION = "2026-07-19-v4"

DIMENSION_COLUMNS = [
    "exploration_score",
    "integration_score",
    "breadth_score",
    "adaptation_score",
]

DIMENSION_LABELS = [
    "exploration",
    "integration",
    "breadth",
    "adaptation",
]


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

ARCHETYPES_PATH = OUT / "organisational_semantic_archetypes.csv"
CONTRIBUTIONS_PATH = OUT / "community_contributions.csv"

PAIRS_PATH = OUT / "company_differentiation_pairs.csv"
OVERALL_MATRIX_PATH = OUT / "company_differentiation_matrix.csv"
DIMENSION_MATRIX_PATH = OUT / "company_dimension_distance_matrix.csv"
JS_MATRIX_PATH = OUT / "company_community_js_matrix.csv"
TV_MATRIX_PATH = OUT / "company_community_tv_matrix.csv"
COSINE_MATRIX_PATH = OUT / "company_community_cosine_matrix.csv"
RANKINGS_PATH = OUT / "company_differentiation_rankings.csv"
METADATA_PATH = OUT / "company_differentiation_metadata.json"


def section(title: str) -> None:
    line = "=" * 78
    print(f"\n{line}\n{title}\n{line}")


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")


def clean_text(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def safe_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator <= 0:
        return np.nan
    return float(np.dot(a, b) / denominator)


def jensen_shannon_divergence(
    p: np.ndarray,
    q: np.ndarray,
) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    p = np.clip(p, 0.0, None)
    q = np.clip(q, 0.0, None)

    if p.sum() <= 0 or q.sum() <= 0:
        return np.nan

    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl_divergence(x: np.ndarray, y: np.ndarray) -> float:
        mask = x > 0
        return float(
            np.sum(x[mask] * np.log2(x[mask] / y[mask]))
        )

    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def total_variation_distance(
    p: np.ndarray,
    q: np.ndarray,
) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    if p.sum() <= 0 or q.sum() <= 0:
        return np.nan

    p = p / p.sum()
    q = q / q.sum()

    return float(0.5 * np.abs(p - q).sum())


def minmax_scale(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    minimum = values.min()
    maximum = values.max()

    if not np.isfinite(minimum) or not np.isfinite(maximum):
        return pd.Series(np.nan, index=series.index)

    if math.isclose(minimum, maximum):
        return pd.Series(0.0, index=series.index)

    return (values - minimum) / (maximum - minimum)


def dominant_contrast(
    company_a: str,
    company_b: str,
    row_a: pd.Series,
    row_b: pd.Series,
) -> tuple[str, float, str]:
    differences = {
        label: abs(
            float(row_a[column]) - float(row_b[column])
        )
        for label, column in zip(
            DIMENSION_LABELS,
            DIMENSION_COLUMNS,
        )
    }

    largest_dimension = max(differences, key=differences.get)
    largest_difference = differences[largest_dimension]

    a_value = float(
        row_a[f"{largest_dimension}_score"]
    )
    b_value = float(
        row_b[f"{largest_dimension}_score"]
    )

    if a_value > b_value:
        direction = f"{company_a} higher"
    elif b_value > a_value:
        direction = f"{company_b} higher"
    else:
        direction = "equal"

    return largest_dimension, largest_difference, direction


def build_square_matrix(
    companies: list[str],
    pair_table: pd.DataFrame,
    value_column: str,
    diagonal_value: float,
) -> pd.DataFrame:
    matrix = pd.DataFrame(
        diagonal_value,
        index=companies,
        columns=companies,
        dtype=float,
    )

    for row in pair_table.itertuples(index=False):
        company_a = row.company_a
        company_b = row.company_b
        value = getattr(row, value_column)

        matrix.loc[company_a, company_b] = value
        matrix.loc[company_b, company_a] = value

    matrix.index.name = "company"
    return matrix.reset_index()


def relative_difference_label(rank: int, pair_count: int) -> str:
    """
    Assign explicitly relative labels from the observed pairwise ranking.

    The labels describe position within the six company pairs and must not be
    interpreted as absolute industry-wide categories.
    """
    if rank == 1:
        return "highest relative differentiation"
    if rank == pair_count:
        return "lowest relative differentiation"

    upper_cutoff = max(2, math.ceil(pair_count / 3))
    lower_cutoff = max(upper_cutoff + 1, math.ceil(2 * pair_count / 3))

    if rank <= upper_cutoff:
        return "high relative differentiation"
    if rank <= lower_cutoff:
        return "moderate relative differentiation"
    return "low relative differentiation"



def resolve_company_probability(
    contributions: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    """
    Return a contribution table containing a canonical company_probability
    column.

    The WP8b output schema may differ slightly across script versions. This
    resolver first accepts known probability/proportion names, then derives
    probabilities from count columns when available, and finally normalises a
    non-negative company-community mass column within company.
    """
    frame = contributions.copy()

    direct_candidates = [
        "company_probability",
        "company_community_probability",
        "company_proportion",
        "company_share",
        "community_probability_within_company",
        "within_company_probability",
        "p_company_community",
    ]

    for column in direct_candidates:
        if column in frame.columns:
            frame["company_probability"] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )
            return frame, f"direct:{column}"

    count_candidates = [
        "company_community_song_count",
        "company_community_count",
        "community_song_count",
        "song_count",
        "n_songs",
        "count",
    ]
    total_candidates = [
        "company_song_count",
        "company_total_song_count",
        "total_company_songs",
        "company_total",
    ]

    count_column = next(
        (column for column in count_candidates if column in frame.columns),
        None,
    )
    total_column = next(
        (column for column in total_candidates if column in frame.columns),
        None,
    )

    if count_column is not None and total_column is not None:
        counts = pd.to_numeric(frame[count_column], errors="coerce")
        totals = pd.to_numeric(frame[total_column], errors="coerce")
        frame["company_probability"] = np.where(
            totals > 0,
            counts / totals,
            np.nan,
        )
        return frame, f"derived:{count_column}/{total_column}"

    if count_column is not None:
        counts = (
            pd.to_numeric(frame[count_column], errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0)
        )
        frame["_probability_mass"] = counts
        denominators = frame.groupby(
            "company",
            observed=True,
        )["_probability_mass"].transform("sum")
        frame["company_probability"] = np.where(
            denominators > 0,
            frame["_probability_mass"] / denominators,
            np.nan,
        )
        frame = frame.drop(columns="_probability_mass")
        return frame, f"normalised_count:{count_column}"

    mass_candidates = [
        "weighted_contribution",
        "absolute_contribution",
        "contribution_magnitude",
        "community_weight",
    ]

    mass_column = next(
        (column for column in mass_candidates if column in frame.columns),
        None,
    )

    if mass_column is not None:
        mass = (
            pd.to_numeric(frame[mass_column], errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0)
        )
        frame["_probability_mass"] = mass
        denominators = frame.groupby(
            "company",
            observed=True,
        )["_probability_mass"].transform("sum")
        frame["company_probability"] = np.where(
            denominators > 0,
            frame["_probability_mass"] / denominators,
            np.nan,
        )
        frame = frame.drop(columns="_probability_mass")
        return frame, f"normalised_mass:{mass_column}"

    raise KeyError(
        "Could not resolve a within-company community probability from "
        "community_contributions.csv. Available columns are: "
        + ", ".join(map(str, frame.columns))
    )

def main() -> None:
    section("WP8e : COMPANY DIFFERENTIATION MATRIX")
    print(f"Script version          : {SCRIPT_VERSION}")
    print(f"Project root            : {ROOT}")
    print(f"Archetype input         : {ARCHETYPES_PATH}")
    print(f"Community input         : {CONTRIBUTIONS_PATH}")

    require(ARCHETYPES_PATH)
    require(CONTRIBUTIONS_PATH)

    archetypes = pd.read_csv(ARCHETYPES_PATH)
    contributions = pd.read_csv(CONTRIBUTIONS_PATH)

    archetypes["company"] = archetypes["company"].map(clean_text)
    contributions["company"] = contributions["company"].map(clean_text)

    missing_archetype_columns = [
        column
        for column in [
            "company",
            "semantic_archetype",
            "dominance_type",
            "positive_dominant_dimensions",
            *DIMENSION_COLUMNS,
        ]
        if column not in archetypes.columns
    ]
    if missing_archetype_columns:
        raise KeyError(
            "Missing archetype columns: "
            + ", ".join(missing_archetype_columns)
        )

    required_contribution_columns = [
        "company",
        "community",
    ]
    missing_contribution_columns = [
        column
        for column in required_contribution_columns
        if column not in contributions.columns
    ]
    if missing_contribution_columns:
        raise KeyError(
            "Missing community-contribution identifier columns: "
            + ", ".join(missing_contribution_columns)
        )

    contributions, probability_source = resolve_company_probability(
        contributions
    )

    invalid_probability_rows = contributions[
        "company_probability"
    ].isna().sum()
    if invalid_probability_rows:
        raise ValueError(
            "Resolved company probabilities contain "
            f"{invalid_probability_rows} missing values."
        )

    probability_sums = contributions.groupby(
        "company",
        observed=True,
    )["company_probability"].sum()

    if not np.allclose(
        probability_sums.to_numpy(dtype=float),
        1.0,
        atol=1e-6,
    ):
        contributions["company_probability"] = contributions[
            "company_probability"
        ] / contributions.groupby(
            "company",
            observed=True,
        )["company_probability"].transform("sum")

    print(f"Probability source      : {probability_source}")

    companies = sorted(archetypes["company"].unique())
    communities = sorted(
        contributions["community"].unique(),
        key=lambda value: str(value),
    )

    section("PREPARING COMPANY-COMMUNITY PROFILES")

    full_grid = pd.MultiIndex.from_product(
        [companies, communities],
        names=["company", "community"],
    ).to_frame(index=False)

    community_profiles = full_grid.merge(
        contributions[
            [
                "company",
                "community",
                "company_probability",
            ]
        ],
        on=["company", "community"],
        how="left",
        validate="one_to_one",
    )

    community_profiles["company_probability"] = (
        community_profiles["company_probability"]
        .fillna(0.0)
        .astype(float)
    )

    probability_matrix = (
        community_profiles.pivot(
            index="company",
            columns="community",
            values="company_probability",
        )
        .reindex(index=companies, columns=communities)
        .fillna(0.0)
    )

    section("COMPUTING PAIRWISE DIFFERENTIATION")

    archetype_lookup = archetypes.set_index("company")
    records: list[dict[str, object]] = []

    for company_a, company_b in combinations(companies, 2):
        row_a = archetype_lookup.loc[company_a]
        row_b = archetype_lookup.loc[company_b]

        vector_a = row_a[DIMENSION_COLUMNS].to_numpy(dtype=float)
        vector_b = row_b[DIMENSION_COLUMNS].to_numpy(dtype=float)

        dimension_euclidean = float(
            np.linalg.norm(vector_a - vector_b)
        )
        dimension_manhattan = float(
            np.abs(vector_a - vector_b).sum()
        )
        dimension_cosine = safe_cosine_similarity(
            vector_a,
            vector_b,
        )

        probability_a = probability_matrix.loc[
            company_a
        ].to_numpy(dtype=float)
        probability_b = probability_matrix.loc[
            company_b
        ].to_numpy(dtype=float)

        community_js = jensen_shannon_divergence(
            probability_a,
            probability_b,
        )
        community_tv = total_variation_distance(
            probability_a,
            probability_b,
        )
        community_cosine = safe_cosine_similarity(
            probability_a,
            probability_b,
        )

        contrast_dimension, contrast_magnitude, contrast_direction = (
            dominant_contrast(
                company_a,
                company_b,
                row_a,
                row_b,
            )
        )

        same_archetype = (
            clean_text(row_a["semantic_archetype"])
            == clean_text(row_b["semantic_archetype"])
        )
        same_dominance_type = (
            clean_text(row_a["dominance_type"])
            == clean_text(row_b["dominance_type"])
        )
        same_dominant_dimensions = (
            clean_text(row_a["positive_dominant_dimensions"])
            == clean_text(row_b["positive_dominant_dimensions"])
        )

        records.append(
            {
                "company_a": company_a,
                "company_b": company_b,
                "archetype_a": clean_text(
                    row_a["semantic_archetype"]
                ),
                "archetype_b": clean_text(
                    row_b["semantic_archetype"]
                ),
                "dimension_euclidean_distance":
                    dimension_euclidean,
                "dimension_manhattan_distance":
                    dimension_manhattan,
                "dimension_cosine_similarity":
                    dimension_cosine,
                "community_js_divergence": community_js,
                "community_total_variation": community_tv,
                "community_cosine_similarity":
                    community_cosine,
                "same_archetype": same_archetype,
                "same_dominance_type": same_dominance_type,
                "same_dominant_dimensions":
                    same_dominant_dimensions,
                "largest_dimension_contrast":
                    contrast_dimension,
                "largest_dimension_difference":
                    contrast_magnitude,
                "largest_dimension_direction":
                    contrast_direction,
            }
        )

    pairs = pd.DataFrame(records)

    pairs["scaled_dimension_distance"] = minmax_scale(
        pairs["dimension_euclidean_distance"]
    )
    pairs["scaled_community_js"] = minmax_scale(
        pairs["community_js_divergence"]
    )
    pairs["scaled_community_tv"] = minmax_scale(
        pairs["community_total_variation"]
    )

    pairs["archetype_structure_difference"] = (
        (
            (~pairs["same_archetype"]).astype(float)
            + (~pairs["same_dominance_type"]).astype(float)
            + (~pairs["same_dominant_dimensions"]).astype(float)
        )
        / 3.0
    )

    pairs["overall_differentiation_index"] = (
        0.50 * pairs["scaled_dimension_distance"]
        + 0.50 * pairs["scaled_community_js"]
    )

    pairs["differentiation_rank"] = (
        pairs["overall_differentiation_index"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )

    pair_count = len(pairs)
    pairs["overall_difference_level"] = pairs[
        "differentiation_rank"
    ].map(
        lambda rank: relative_difference_label(
            int(rank),
            pair_count,
        )
    )

    pairs["pairwise_interpretation"] = pairs.apply(
        lambda row: (
            f"{row['company_a']} and {row['company_b']} show "
            f"{row['overall_difference_level']} within the observed "
            f"company set. Their largest latent contrast is "
            f"{row['largest_dimension_contrast']} "
            f"({row['largest_dimension_direction']}; absolute difference "
            f"{row['largest_dimension_difference']:.3f}), while their "
            f"community distributions have JS divergence "
            f"{row['community_js_divergence']:.3f}. Total variation "
            f"({row['community_total_variation']:.3f}) and community cosine "
            f"similarity ({row['community_cosine_similarity']:.3f}) are "
            f"reported as supporting diagnostics and do not enter the "
            f"composite index."
        ),
        axis=1,
    )

    pairs = pairs.sort_values(
        [
            "overall_differentiation_index",
            "company_a",
            "company_b",
        ],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    section("BUILDING MATRICES")

    overall_matrix = build_square_matrix(
        companies,
        pairs,
        "overall_differentiation_index",
        0.0,
    )
    dimension_matrix = build_square_matrix(
        companies,
        pairs,
        "dimension_euclidean_distance",
        0.0,
    )
    js_matrix = build_square_matrix(
        companies,
        pairs,
        "community_js_divergence",
        0.0,
    )
    tv_matrix = build_square_matrix(
        companies,
        pairs,
        "community_total_variation",
        0.0,
    )
    cosine_matrix = build_square_matrix(
        companies,
        pairs,
        "community_cosine_similarity",
        1.0,
    )

    rankings = pairs[
        [
            "differentiation_rank",
            "company_a",
            "company_b",
            "archetype_a",
            "archetype_b",
            "overall_differentiation_index",
            "overall_difference_level",
            "largest_dimension_contrast",
            "largest_dimension_difference",
            "largest_dimension_direction",
            "dimension_euclidean_distance",
            "community_js_divergence",
            "community_total_variation",
            "community_cosine_similarity",
            "pairwise_interpretation",
        ]
    ].copy()

    pairs.to_csv(
        PAIRS_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    overall_matrix.to_csv(
        OVERALL_MATRIX_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    dimension_matrix.to_csv(
        DIMENSION_MATRIX_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    js_matrix.to_csv(
        JS_MATRIX_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    tv_matrix.to_csv(
        TV_MATRIX_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    cosine_matrix.to_csv(
        COSINE_MATRIX_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    rankings.to_csv(
        RANKINGS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "script": Path(__file__).name,
        "script_version": SCRIPT_VERSION,
        "analysis_name": "WP8e Company Differentiation Matrix",
        "companies": companies,
        "community_count": len(communities),
        "community_probability_source": probability_source,
        "latent_dimensions": DIMENSION_LABELS,
        "overall_differentiation_index": {
            "formula": (
                "0.50 * scaled latent-dimension Euclidean distance + "
                "0.50 * scaled community JS divergence"
            ),
            "weights": {
                "latent_dimension_distance": 0.50,
                "community_js_divergence": 0.50,
            },
            "excluded_from_composite_index": {
                "community_total_variation": (
                    "Reported as a supporting diagnostic because it measures "
                    "the same underlying community-distribution difference as "
                    "Jensen-Shannon divergence."
                ),
                "community_cosine_similarity": (
                    "Reported as a supporting similarity diagnostic."
                ),
                "archetype_structure_difference": (
                    "Reported descriptively only because archetypes and "
                    "dominance patterns are derived from the latent dimensions."
                ),
            },
            "scaling": (
                "Each component is min-max scaled across the six observed "
                "company pairs before equal weighting."
            ),
            "interpretation": (
                "The index is a relative within-sample differentiation "
                "measure combining latent organisational configuration and "
                "semantic-community composition. It must not be interpreted "
                "as an absolute industry-wide distance."
            ),
        },
        "relative_difference_labels": {
            "highest relative differentiation": (
                "The top-ranked pair within the observed company set."
            ),
            "high relative differentiation": (
                "A pair in the upper part of the observed ranking."
            ),
            "moderate relative differentiation": (
                "A pair in the middle part of the observed ranking."
            ),
            "low relative differentiation": (
                "A pair in the lower part of the observed ranking."
            ),
            "lowest relative differentiation": (
                "The bottom-ranked pair within the observed company set."
            ),
        },
        "archetype_structure_difference": {
            "components": [
                "different semantic archetype",
                "different dominance type",
                "different positive dominant dimensions",
            ],
            "aggregation": (
                "Mean of three binary disagreement indicators."
            ),
        },
        "pair_count": int(len(pairs)),
        "outputs": {
            "pairwise_results": str(PAIRS_PATH),
            "overall_matrix": str(OVERALL_MATRIX_PATH),
            "dimension_matrix": str(DIMENSION_MATRIX_PATH),
            "community_js_matrix": str(JS_MATRIX_PATH),
            "community_tv_matrix": str(TV_MATRIX_PATH),
            "community_cosine_matrix": str(COSINE_MATRIX_PATH),
            "rankings": str(RANKINGS_PATH),
        },
    }

    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    section("PAIRWISE DIFFERENTIATION RANKING")

    display_columns = [
        "differentiation_rank",
        "company_a",
        "company_b",
        "overall_differentiation_index",
        "overall_difference_level",
        "largest_dimension_contrast",
        "largest_dimension_difference",
        "community_js_divergence",
        "community_total_variation",
    ]

    print(
        rankings[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    section("PAIRWISE INTERPRETATIONS")

    for row in rankings.itertuples(index=False):
        print(
            f"\n{row.company_a} vs {row.company_b} "
            f"(rank {row.differentiation_rank})"
        )
        print(row.pairwise_interpretation)

    section("OUTPUTS")
    print(f"Pairwise results        : {PAIRS_PATH}")
    print(f"Overall matrix          : {OVERALL_MATRIX_PATH}")
    print(f"Dimension matrix        : {DIMENSION_MATRIX_PATH}")
    print(f"Community JS matrix     : {JS_MATRIX_PATH}")
    print(f"Community TV matrix     : {TV_MATRIX_PATH} (diagnostic)")
    print(f"Community cosine matrix : {COSINE_MATRIX_PATH} (diagnostic)")
    print(f"Rankings                : {RANKINGS_PATH}")
    print(f"Metadata                : {METADATA_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"\nWP8e failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        raise
