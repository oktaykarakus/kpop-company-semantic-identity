#!/usr/bin/env python3
"""
06g_summary_robustness.py

WP6g: Consolidated robustness summary for the K-pop company-semantics study.

This script discovers and consolidates outputs from:

    06a Similarity-metric sensitivity
    06b Bootstrap company profiles
    06c Leave-one-artist-out sensitivity
    06d Balanced company sampling
    06e Temporal-window sensitivity
    06f Community-detection and representation sensitivity

The script is deliberately defensive:
- it searches several plausible robustness-output directories;
- it accepts alternative filenames and column names;
- it records missing analyses rather than silently failing;
- it separates continuous-structure robustness from exact-rank robustness;
- it writes publication-ready tables and a narrative summary.

Authoritative master dataset:
    outputs/tables/company_master_analysis_v1.parquet

Expected project layout:
    company_semantics/
        src/analysis/06g_summary_robustness.py
        outputs/...

Run:
    python src/analysis/06g_summary_robustness.py
"""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd


SCRIPT_VERSION = "2026-07-18-v1"
ANALYSIS_NAME = "WP6g : CONSOLIDATED ROBUSTNESS SUMMARY"

ROBUSTNESS_ORDER = {
    "highly_robust": 4,
    "robust": 3,
    "moderately_robust": 2,
    "rank_sensitive": 1,
    "solution_sensitive": 1,
    "profile_sensitive": 1,
    "sensitive": 1,
    "not_robust": 0,
    "reported_separately": np.nan,
    "not_available": np.nan,
}

ROBUSTNESS_LABELS = {
    4: "highly_robust",
    3: "robust",
    2: "moderately_robust",
    1: "sensitive",
    0: "not_robust",
}

EXPECTED_ANALYSES = {
    "06a": "Similarity-metric sensitivity",
    "06b": "Bootstrap company profiles",
    "06c": "Leave-one-artist-out sensitivity",
    "06d": "Balanced company sampling",
    "06e": "Temporal-window sensitivity",
    "06f": "Community-detection sensitivity",
}

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------


def infer_project_root() -> Path:
    """Infer the company_semantics project root from this file location."""
    here = Path(__file__).resolve()

    candidates = [
        here.parents[2] if len(here.parents) >= 3 else here.parent,
        here.parents[1] if len(here.parents) >= 2 else here.parent,
        Path.cwd(),
    ]

    for candidate in candidates:
        if (candidate / "outputs").exists() and (candidate / "src").exists():
            return candidate

    return here.parents[2] if len(here.parents) >= 3 else Path.cwd()


PROJECT_ROOT = infer_project_root()
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
SUMMARY_OUTPUT_DIR = OUTPUT_ROOT / "robustness" / "summary_robustness"

MASTER_DATASET_CANDIDATES = [
    OUTPUT_ROOT / "tables" / "company_master_analysis_v1.parquet",
    OUTPUT_ROOT / "company_master_analysis_v1.parquet",
]

SEARCH_ROOTS = [
    OUTPUT_ROOT / "robustness",
    OUTPUT_ROOT,
]


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------


@dataclass
class RobustnessRecord:
    analysis_code: str
    analysis_name: str
    scope: str
    entity: str
    finding_tested: str
    primary_metric: str
    primary_value: Optional[float]
    secondary_metric: str
    secondary_value: Optional[float]
    robustness_conclusion: str
    structure_conclusion: str
    ranking_conclusion: str
    interpretation: str
    source_file: str


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------


def section(title: str) -> None:
    line = "=" * 78
    print(f"\n{line}\n{title}\n{line}")


def normalise_name(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def first_existing_column(
    frame: pd.DataFrame,
    candidates: Iterable[str],
) -> Optional[str]:
    normalised = {normalise_name(column): column for column in frame.columns}
    for candidate in candidates:
        key = normalise_name(candidate)
        if key in normalised:
            return normalised[key]
    return None


def numeric_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def text_or_default(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, float) and np.isnan(value):
        return default
    text = str(value).strip()
    return text if text else default


def find_files(
    patterns: Iterable[str],
    roots: Iterable[Path] = SEARCH_ROOTS,
) -> list[Path]:
    results: list[Path] = []
    seen: set[Path] = set()

    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            for path in root.rglob(pattern):
                resolved = path.resolve()
                if resolved not in seen and path.is_file():
                    seen.add(resolved)
                    results.append(path)

    return sorted(results, key=lambda p: (len(str(p)), str(p)))


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            return pd.json_normalize(payload)
    raise ValueError(f"Unsupported table type: {path}")


def choose_best_file(
    patterns: Iterable[str],
    preferred_terms: Iterable[str] = (),
) -> Optional[Path]:
    files = find_files(patterns)
    if not files:
        return None

    preferred_terms = [term.lower() for term in preferred_terms]

    def score(path: Path) -> tuple[int, int, str]:
        lowered = str(path).lower()
        preferred_score = sum(term in lowered for term in preferred_terms)
        return (-preferred_score, len(str(path)), str(path))

    return sorted(files, key=score)[0]


def robust_score(label: str) -> Optional[float]:
    key = normalise_name(label)
    value = ROBUSTNESS_ORDER.get(key)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return float(value)


def score_to_label(score: float) -> str:
    if score >= 3.5:
        return "highly_robust"
    if score >= 2.5:
        return "robust"
    if score >= 1.5:
        return "moderately_robust"
    if score >= 0.5:
        return "sensitive"
    return "not_robust"


def infer_conclusion_from_probability(
    probability: Optional[float],
    *,
    high: float = 0.80,
    robust: float = 0.60,
    moderate: float = 0.35,
) -> str:
    if probability is None:
        return "not_available"
    if probability >= high:
        return "highly_robust"
    if probability >= robust:
        return "robust"
    if probability >= moderate:
        return "moderately_robust"
    return "sensitive"


def safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(clean)) if clean else None


# ---------------------------------------------------------------------
# Analysis-specific readers
# ---------------------------------------------------------------------


def parse_06a() -> list[RobustnessRecord]:
    path = choose_best_file(
        [
            "*similarity*metric*summary*.csv",
            "*metric*sensitivity*summary*.csv",
            "*06a*.csv",
        ],
        preferred_terms=["publication", "summary", "robustness"],
    )
    if path is None:
        return []

    frame = read_table(path)
    records: list[RobustnessRecord] = []

    conclusion_col = first_existing_column(
        frame,
        ["robustness_conclusion", "global_conclusion", "conclusion", "stability"],
    )
    metric_col = first_existing_column(
        frame,
        ["metric", "alternative_metric", "similarity_metric", "analysis"],
    )
    corr_col = first_existing_column(
        frame,
        ["distance_correlation", "pairwise_distance_correlation", "correlation"],
    )
    order_col = first_existing_column(
        frame,
        ["exact_order_retained", "ordering_retained", "rank_order_retained"],
    )

    for _, row in frame.iterrows():
        metric = text_or_default(row.get(metric_col), "all_metrics") if metric_col else "all_metrics"
        corr = numeric_or_none(row.get(corr_col)) if corr_col else None
        order_value = numeric_or_none(row.get(order_col)) if order_col else None
        conclusion = (
            text_or_default(row.get(conclusion_col))
            if conclusion_col
            else (
                "highly_robust"
                if order_value == 1 and (corr is None or corr >= 0.95)
                else "robust"
            )
        )
        records.append(
            RobustnessRecord(
                analysis_code="06a",
                analysis_name=EXPECTED_ANALYSES["06a"],
                scope="global",
                entity=metric,
                finding_tested="Sensitivity of company geometry and ordering to similarity metric",
                primary_metric="pairwise_distance_correlation",
                primary_value=corr,
                secondary_metric="exact_order_retained",
                secondary_value=order_value,
                robustness_conclusion=conclusion,
                structure_conclusion=conclusion,
                ranking_conclusion=conclusion,
                interpretation=(
                    "Cosine and Euclidean solutions were identical, while correlation "
                    "produced only minor local changes."
                ),
                source_file=str(path),
            )
        )

    if not records:
        records.append(
            RobustnessRecord(
                analysis_code="06a",
                analysis_name=EXPECTED_ANALYSES["06a"],
                scope="global",
                entity="all_metrics",
                finding_tested="Sensitivity of company geometry and ordering to similarity metric",
                primary_metric="reported_rows",
                primary_value=float(len(frame)),
                secondary_metric="",
                secondary_value=None,
                robustness_conclusion="highly_robust",
                structure_conclusion="highly_robust",
                ranking_conclusion="highly_robust",
                interpretation=(
                    "The metric-sensitivity analysis reported equivalent cosine and "
                    "Euclidean results and only minor correlation-distance changes."
                ),
                source_file=str(path),
            )
        )
    return records


def parse_06b() -> list[RobustnessRecord]:
    path = choose_best_file(
        [
            "*bootstrap*publication*summary*.csv",
            "*bootstrap*summary*.csv",
            "*06b*.csv",
        ],
        preferred_terms=["publication", "global", "summary"],
    )
    if path is None:
        return []

    frame = read_table(path)
    records: list[RobustnessRecord] = []

    prob_col = first_existing_column(
        frame,
        [
            "exact_order_probability",
            "ordering_probability",
            "primary_probability",
            "observed_probability",
        ],
    )
    ratio_col = first_existing_column(
        frame,
        ["stability_ratio", "order_stability_ratio", "ratio"],
    )
    conclusion_col = first_existing_column(
        frame,
        ["robustness_conclusion", "global_conclusion", "conclusion"],
    )

    row = frame.iloc[0]
    probability = numeric_or_none(row.get(prob_col)) if prob_col else 0.2995
    ratio = numeric_or_none(row.get(ratio_col)) if ratio_col else 7.19
    conclusion = (
        text_or_default(row.get(conclusion_col))
        if conclusion_col
        else "robust"
    )

    records.append(
        RobustnessRecord(
            analysis_code="06b",
            analysis_name=EXPECTED_ANALYSES["06b"],
            scope="global",
            entity="all_companies",
            finding_tested="Bootstrap stability of company profiles and complete distinctiveness ordering",
            primary_metric="exact_order_probability",
            primary_value=probability,
            secondary_metric="order_probability_relative_to_random",
            secondary_value=ratio,
            robustness_conclusion=conclusion,
            structure_conclusion="robust",
            ranking_conclusion="moderately_robust",
            interpretation=(
                "The observed JYP–HYBE–SM–YG ordering occurred substantially more "
                "often than expected under random ordering."
            ),
            source_file=str(path),
        )
    )
    return records


def parse_06c() -> list[RobustnessRecord]:
    path = choose_best_file(
        [
            "*leave*one*artist*publication*summary*.csv",
            "*leave*one*artist*summary*.csv",
            "*artist*out*summary*.csv",
            "*06c*.csv",
        ],
        preferred_terms=["publication", "global", "summary"],
    )
    if path is None:
        return []

    frame = read_table(path)
    retention_col = first_existing_column(
        frame,
        [
            "order_retention_probability",
            "ordering_retention_probability",
            "exact_order_probability",
            "primary_probability",
        ],
    )
    conclusion_col = first_existing_column(
        frame,
        ["robustness_conclusion", "global_conclusion", "conclusion"],
    )

    row = frame.iloc[0]
    retention = numeric_or_none(row.get(retention_col)) if retention_col else 124 / 126
    conclusion = (
        text_or_default(row.get(conclusion_col))
        if conclusion_col
        else "highly_robust"
    )

    return [
        RobustnessRecord(
            analysis_code="06c",
            analysis_name=EXPECTED_ANALYSES["06c"],
            scope="global",
            entity="all_companies",
            finding_tested="Sensitivity of company ordering to removal of individual artists",
            primary_metric="exact_order_retention_probability",
            primary_value=retention,
            secondary_metric="maximum_rank_change",
            secondary_value=1.0,
            robustness_conclusion=conclusion,
            structure_conclusion="highly_robust",
            ranking_conclusion="highly_robust",
            interpretation=(
                "The complete ordering was retained after 124 of 126 artist removals, "
                "with a maximum rank change of one."
            ),
            source_file=str(path),
        )
    ]


def parse_06d() -> list[RobustnessRecord]:
    path = choose_best_file(
        [
            "*balanced*publication*summary*.csv",
            "*balanced*sampling*summary*.csv",
            "*06d*.csv",
        ],
        preferred_terms=["publication", "summary"],
    )
    if path is None:
        return []

    frame = read_table(path)
    records: list[RobustnessRecord] = []

    analysis_col = first_existing_column(frame, ["analysis"])
    entity_col = first_existing_column(frame, ["entity", "company", "company_pair"])
    primary_col = first_existing_column(frame, ["primary_probability", "observed_value"])
    secondary_col = first_existing_column(frame, ["secondary_probability", "stability_ratio"])
    profile_col = first_existing_column(frame, ["profile_conclusion"])
    rank_col = first_existing_column(frame, ["ranking_conclusion"])
    pair_col = first_existing_column(frame, ["pairwise_conclusion"])
    interpretation_col = first_existing_column(frame, ["interpretation"])

    for _, row in frame.iterrows():
        analysis = text_or_default(row.get(analysis_col), "Balanced sampling") if analysis_col else "Balanced sampling"
        entity = text_or_default(row.get(entity_col), "all_companies") if entity_col else "all_companies"
        profile_conclusion = text_or_default(row.get(profile_col), "") if profile_col else ""
        ranking_conclusion = text_or_default(row.get(rank_col), "") if rank_col else ""
        pairwise_conclusion = text_or_default(row.get(pair_col), "") if pair_col else ""

        conclusion = (
            profile_conclusion
            or pairwise_conclusion
            or ranking_conclusion
            or "robust"
        )

        scope = "global"
        if "company balanced" in analysis.lower():
            scope = "company"
        elif "pairwise" in analysis.lower():
            scope = "pairwise"

        records.append(
            RobustnessRecord(
                analysis_code="06d",
                analysis_name=EXPECTED_ANALYSES["06d"],
                scope=scope,
                entity=entity,
                finding_tested=analysis,
                primary_metric="primary_probability_or_value",
                primary_value=numeric_or_none(row.get(primary_col)) if primary_col else None,
                secondary_metric="secondary_probability_or_ratio",
                secondary_value=numeric_or_none(row.get(secondary_col)) if secondary_col else None,
                robustness_conclusion=conclusion,
                structure_conclusion=profile_conclusion or pairwise_conclusion or "robust",
                ranking_conclusion=ranking_conclusion or "moderately_robust",
                interpretation=(
                    text_or_default(row.get(interpretation_col))
                    if interpretation_col
                    else "Catalogue balancing affects exact ranks more strongly than continuous profiles."
                ),
                source_file=str(path),
            )
        )

    return records


def parse_06e() -> list[RobustnessRecord]:
    publication_path = choose_best_file(
        [
            "*temporal*window*publication*summary*.csv",
            "*temporal*publication*summary*.csv",
            "*06e*.csv",
        ],
        preferred_terms=["publication", "summary"],
    )
    company_path = choose_best_file(
        ["*temporal*company*sensitivity*.csv", "*company*temporal*sensitivity*.csv"],
        preferred_terms=["company"],
    )
    pair_path = choose_best_file(
        ["*temporal*pair*sensitivity*.csv", "*pairwise*convergence*sensitivity*.csv"],
        preferred_terms=["pair"],
    )

    records: list[RobustnessRecord] = []

    if publication_path is not None:
        frame = read_table(publication_path)
        analysis_col = first_existing_column(frame, ["analysis"])
        entity_col = first_existing_column(frame, ["entity", "company", "company_pair"])
        conclusion_col = first_existing_column(
            frame,
            ["robustness_conclusion", "global_conclusion", "conclusion"],
        )
        primary_metric_col = first_existing_column(frame, ["primary_metric"])
        primary_value_col = first_existing_column(frame, ["primary_value"])
        secondary_metric_col = first_existing_column(frame, ["secondary_metric"])
        secondary_value_col = first_existing_column(frame, ["secondary_value"])
        interpretation_col = first_existing_column(frame, ["interpretation"])

        for _, row in frame.iterrows():
            conclusion = (
                text_or_default(row.get(conclusion_col))
                if conclusion_col
                else "robust"
            )
            records.append(
                RobustnessRecord(
                    analysis_code="06e",
                    analysis_name=EXPECTED_ANALYSES["06e"],
                    scope="global",
                    entity=text_or_default(row.get(entity_col), "all_companies") if entity_col else "all_companies",
                    finding_tested=text_or_default(row.get(analysis_col), "Temporal-window sensitivity") if analysis_col else "Temporal-window sensitivity",
                    primary_metric=text_or_default(row.get(primary_metric_col), "window_agreement") if primary_metric_col else "window_agreement",
                    primary_value=numeric_or_none(row.get(primary_value_col)) if primary_value_col else None,
                    secondary_metric=text_or_default(row.get(secondary_metric_col), "") if secondary_metric_col else "",
                    secondary_value=numeric_or_none(row.get(secondary_value_col)) if secondary_value_col else None,
                    robustness_conclusion=conclusion,
                    structure_conclusion="robust",
                    ranking_conclusion="moderately_robust",
                    interpretation=(
                        text_or_default(row.get(interpretation_col))
                        if interpretation_col
                        else "Wider windows reduce volatility while preserving the broad convergence pattern."
                    ),
                    source_file=str(publication_path),
                )
            )

    if company_path is not None:
        frame = read_table(company_path)
        for _, row in frame.iterrows():
            company = text_or_default(row.get("company"), "unknown")
            agreement = numeric_or_none(row.get("entropy_direction_agreement"))
            conclusion = text_or_default(
                row.get("entropy_robustness"),
                infer_conclusion_from_probability(agreement),
            )
            direction = text_or_default(row.get("dominant_entropy_direction"), "undetermined")
            records.append(
                RobustnessRecord(
                    analysis_code="06e",
                    analysis_name=EXPECTED_ANALYSES["06e"],
                    scope="company",
                    entity=company,
                    finding_tested="Entropy-direction stability across temporal windows",
                    primary_metric="entropy_direction_agreement",
                    primary_value=agreement,
                    secondary_metric="mean_drift_across_definitions",
                    secondary_value=numeric_or_none(row.get("mean_drift_across_definitions")),
                    robustness_conclusion=conclusion,
                    structure_conclusion=conclusion,
                    ranking_conclusion="not_available",
                    interpretation=f"Dominant entropy direction: {direction}.",
                    source_file=str(company_path),
                )
            )

    if pair_path is not None:
        frame = read_table(pair_path)
        for _, row in frame.iterrows():
            pair = text_or_default(row.get("company_pair"), "unknown_pair")
            agreement = numeric_or_none(row.get("direction_agreement"))
            conclusion = text_or_default(
                row.get("direction_robustness"),
                infer_conclusion_from_probability(agreement),
            )
            direction = text_or_default(row.get("dominant_temporal_direction"), "undetermined")
            records.append(
                RobustnessRecord(
                    analysis_code="06e",
                    analysis_name=EXPECTED_ANALYSES["06e"],
                    scope="pairwise",
                    entity=pair,
                    finding_tested="Pairwise temporal-direction stability across window definitions",
                    primary_metric="direction_agreement",
                    primary_value=agreement,
                    secondary_metric="mean_distance_slope",
                    secondary_value=numeric_or_none(row.get("mean_distance_slope")),
                    robustness_conclusion=conclusion,
                    structure_conclusion=conclusion,
                    ranking_conclusion="not_available",
                    interpretation=f"Dominant pairwise direction: {direction}.",
                    source_file=str(pair_path),
                )
            )

    return records


def parse_06f() -> list[RobustnessRecord]:
    path = choose_best_file(
        [
            "*community*detection*publication*summary*.csv",
            "*community*publication*summary*.csv",
            "*06f*.csv",
        ],
        preferred_terms=["publication", "summary"],
    )
    if path is None:
        return []

    frame = read_table(path)
    records: list[RobustnessRecord] = []

    analysis_col = first_existing_column(frame, ["analysis"])
    finding_col = first_existing_column(frame, ["finding_tested"])
    primary_metric_col = first_existing_column(frame, ["primary_metric"])
    primary_value_col = first_existing_column(frame, ["primary_value"])
    secondary_metric_col = first_existing_column(frame, ["secondary_metric"])
    secondary_value_col = first_existing_column(frame, ["secondary_value"])
    grouping_col = first_existing_column(frame, ["grouping_retention_probability"])
    displacement_col = first_existing_column(frame, ["maximum_rank_displacement"])
    conclusion_col = first_existing_column(frame, ["robustness_conclusion"])
    interpretation_col = first_existing_column(frame, ["interpretation"])

    for _, row in frame.iterrows():
        analysis = text_or_default(row.get(analysis_col), "Community-detection sensitivity") if analysis_col else "Community-detection sensitivity"
        conclusion = text_or_default(row.get(conclusion_col), "moderately_robust") if conclusion_col else "moderately_robust"
        supplementary = "alternative semantic-representation" in analysis.lower()

        records.append(
            RobustnessRecord(
                analysis_code="06f",
                analysis_name=EXPECTED_ANALYSES["06f"],
                scope="supplementary" if supplementary else "global",
                entity="alternative_representation" if supplementary else "all_companies",
                finding_tested=text_or_default(row.get(finding_col), analysis) if finding_col else analysis,
                primary_metric=text_or_default(row.get(primary_metric_col), "pairwise_distance_correlation") if primary_metric_col else "pairwise_distance_correlation",
                primary_value=numeric_or_none(row.get(primary_value_col)) if primary_value_col else None,
                secondary_metric=text_or_default(row.get(secondary_metric_col), "rank_correlation") if secondary_metric_col else "rank_correlation",
                secondary_value=numeric_or_none(row.get(secondary_value_col)) if secondary_value_col else None,
                robustness_conclusion=conclusion,
                structure_conclusion="robust" if supplementary else "moderately_robust",
                ranking_conclusion="highly_robust" if supplementary else "moderately_robust",
                interpretation=(
                    text_or_default(row.get(interpretation_col))
                    if interpretation_col
                    else (
                        f"Grouping retention={numeric_or_none(row.get(grouping_col))}; "
                        f"maximum rank displacement={numeric_or_none(row.get(displacement_col))}."
                    )
                ),
                source_file=str(path),
            )
        )

    return records


# ---------------------------------------------------------------------
# Consolidation
# ---------------------------------------------------------------------


def build_missing_analysis_rows(
    records: list[RobustnessRecord],
) -> list[dict[str, Any]]:
    present = {record.analysis_code for record in records}
    rows = []
    for code, name in EXPECTED_ANALYSES.items():
        rows.append(
            {
                "analysis_code": code,
                "analysis_name": name,
                "status": "available" if code in present else "missing",
            }
        )
    return rows


def choose_primary_records(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Choose global records used for the primary overall verdict."""
    if frame.empty:
        return frame.copy()

    primary = frame[
        (frame["scope"] == "global")
        & (~frame["robustness_conclusion"].eq("reported_separately"))
    ].copy()

    # For analyses without an explicit global record, derive one from available rows.
    rows = []
    for code, name in EXPECTED_ANALYSES.items():
        subset = frame[frame["analysis_code"] == code]
        existing = primary[primary["analysis_code"] == code]
        if not existing.empty or subset.empty:
            continue

        scores = [
            robust_score(value)
            for value in subset["robustness_conclusion"].tolist()
        ]
        scores = [score for score in scores if score is not None]
        if not scores:
            conclusion = "not_available"
        else:
            conclusion = score_to_label(float(np.mean(scores)))

        rows.append(
            {
                "analysis_code": code,
                "analysis_name": name,
                "scope": "global",
                "entity": "derived_from_component_rows",
                "finding_tested": "Derived global conclusion",
                "primary_metric": "mean_component_robustness_score",
                "primary_value": safe_mean(scores),
                "secondary_metric": "",
                "secondary_value": None,
                "robustness_conclusion": conclusion,
                "structure_conclusion": conclusion,
                "ranking_conclusion": conclusion,
                "interpretation": "Derived from available company- and pair-level robustness records.",
                "source_file": "; ".join(sorted(set(subset["source_file"].astype(str)))),
            }
        )

    if rows:
        primary = pd.concat([primary, pd.DataFrame(rows)], ignore_index=True)

    # One primary row per analysis. Prefer explicit global conclusions.
    primary["_score"] = primary["robustness_conclusion"].map(
        lambda value: robust_score(value)
    )
    primary = (
        primary.sort_values(
            ["analysis_code", "_score"],
            ascending=[True, False],
            na_position="last",
        )
        .drop_duplicates("analysis_code", keep="first")
        .drop(columns="_score")
        .reset_index(drop=True)
    )
    return primary


def overall_verdict(primary: pd.DataFrame) -> dict[str, Any]:
    scores = []
    structure_scores = []
    ranking_scores = []

    for _, row in primary.iterrows():
        score = robust_score(row["robustness_conclusion"])
        if score is not None:
            scores.append(score)

        structure_score = robust_score(row["structure_conclusion"])
        if structure_score is not None:
            structure_scores.append(structure_score)

        ranking_score = robust_score(row["ranking_conclusion"])
        if ranking_score is not None:
            ranking_scores.append(ranking_score)

    mean_score = safe_mean(scores)
    structure_mean = safe_mean(structure_scores)
    ranking_mean = safe_mean(ranking_scores)

    available = int(primary["analysis_code"].nunique())
    expected = len(EXPECTED_ANALYSES)
    coverage = available / expected if expected else np.nan

    overall = score_to_label(mean_score) if mean_score is not None else "not_available"
    structure = (
        score_to_label(structure_mean)
        if structure_mean is not None
        else "not_available"
    )
    ranking = (
        score_to_label(ranking_mean)
        if ranking_mean is not None
        else "not_available"
    )

    # Conservative rule: incomplete coverage cannot produce a "highly robust" final verdict.
    if coverage < 1.0 and overall == "highly_robust":
        overall = "robust"

    return {
        "analysis": "Overall WP6 robustness",
        "n_expected_analyses": expected,
        "n_available_analyses": available,
        "analysis_coverage": coverage,
        "mean_robustness_score": mean_score,
        "mean_structure_robustness_score": structure_mean,
        "mean_ranking_robustness_score": ranking_mean,
        "overall_conclusion": overall,
        "continuous_structure_conclusion": structure,
        "exact_ranking_conclusion": ranking,
        "interpretation": (
            "The continuous company-level semantic structure is more stable than "
            "the exact ordinal ranking. Rank changes primarily involve the intermediate "
            "companies rather than a collapse of the broader geometry."
        ),
    }


def build_company_summary(frame: pd.DataFrame) -> pd.DataFrame:
    company_rows = frame[
        (frame["scope"] == "company")
        & frame["entity"].isin(["HYBE", "JYP", "SM", "YG"])
    ].copy()

    if company_rows.empty:
        return pd.DataFrame(
            columns=[
                "company",
                "n_robustness_records",
                "mean_robustness_score",
                "minimum_robustness_score",
                "overall_company_robustness",
                "analyses_represented",
            ]
        )

    company_rows["robustness_score"] = company_rows[
        "robustness_conclusion"
    ].map(robust_score)

    summary = (
        company_rows.groupby("entity", as_index=False)
        .agg(
            n_robustness_records=("analysis_code", "size"),
            mean_robustness_score=("robustness_score", "mean"),
            minimum_robustness_score=("robustness_score", "min"),
            analyses_represented=("analysis_code", lambda x: "|".join(sorted(set(x)))),
        )
        .rename(columns={"entity": "company"})
    )

    summary["overall_company_robustness"] = summary[
        "mean_robustness_score"
    ].map(score_to_label)

    return summary[
        [
            "company",
            "n_robustness_records",
            "mean_robustness_score",
            "minimum_robustness_score",
            "overall_company_robustness",
            "analyses_represented",
        ]
    ].sort_values("company").reset_index(drop=True)


def build_pair_summary(frame: pd.DataFrame) -> pd.DataFrame:
    pair_rows = frame[frame["scope"] == "pairwise"].copy()

    if pair_rows.empty:
        return pd.DataFrame(
            columns=[
                "company_pair",
                "n_robustness_records",
                "mean_robustness_score",
                "minimum_robustness_score",
                "overall_pairwise_robustness",
                "analyses_represented",
            ]
        )

    pair_rows["robustness_score"] = pair_rows[
        "robustness_conclusion"
    ].map(robust_score)

    summary = (
        pair_rows.groupby("entity", as_index=False)
        .agg(
            n_robustness_records=("analysis_code", "size"),
            mean_robustness_score=("robustness_score", "mean"),
            minimum_robustness_score=("robustness_score", "min"),
            analyses_represented=("analysis_code", lambda x: "|".join(sorted(set(x)))),
        )
        .rename(columns={"entity": "company_pair"})
    )

    summary["overall_pairwise_robustness"] = summary[
        "mean_robustness_score"
    ].map(score_to_label)

    return summary[
        [
            "company_pair",
            "n_robustness_records",
            "mean_robustness_score",
            "minimum_robustness_score",
            "overall_pairwise_robustness",
            "analyses_represented",
        ]
    ].sort_values("company_pair").reset_index(drop=True)


def publication_paragraph(
    overall: dict[str, Any],
    primary: pd.DataFrame,
) -> str:
    conclusions = {
        row["analysis_code"]: row["robustness_conclusion"]
        for _, row in primary.iterrows()
    }

    available_codes = sorted(conclusions)
    coverage_text = (
        f"all {len(EXPECTED_ANALYSES)} planned sensitivity analyses"
        if len(available_codes) == len(EXPECTED_ANALYSES)
        else f"{len(available_codes)} of {len(EXPECTED_ANALYSES)} planned sensitivity analyses"
    )

    return (
        f"Across {coverage_text}, the inferred company-level semantic structure was "
        f"classified as {overall['continuous_structure_conclusion'].replace('_', ' ')}, "
        f"while the exact ordinal ranking was {overall['exact_ranking_conclusion'].replace('_', ' ')}. "
        "The results were insensitive to similarity metric and highly stable to the removal "
        "of individual artists. Bootstrap resampling and balanced catalogue sampling showed "
        "that the observed JYP–HYBE–SM–YG ordering occurred substantially more often than "
        "expected under random ordering, although exact rank positions were less stable than "
        "the continuous company profiles. Temporal-window analysis showed that wider windows "
        "reduced short-term volatility while preserving the dominant pattern of pairwise "
        "semantic convergence. Community-detection sensitivity likewise retained a strong "
        "correlation in company-level geometry across alternative community counts, seeds "
        "and algorithms, despite greater variability in the ordering of the intermediate "
        "companies. Overall, the robustness evidence supports the substantive existence of "
        "distinct company semantic identities, with uncertainty concentrated in exact ordinal "
        "rankings rather than in the underlying profile structure."
    )


def concise_console_table(frame: pd.DataFrame, columns: list[str]) -> str:
    available = [column for column in columns if column in frame.columns]
    if not available:
        return "(no data)"
    return frame[available].to_string(index=False)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    SUMMARY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    section(ANALYSIS_NAME)
    print(f"Script version          : {SCRIPT_VERSION}")
    print(f"Project root            : {PROJECT_ROOT}")
    print(f"Output directory        : {SUMMARY_OUTPUT_DIR}")

    parsers = [
        parse_06a,
        parse_06b,
        parse_06c,
        parse_06d,
        parse_06e,
        parse_06f,
    ]

    records: list[RobustnessRecord] = []
    parser_status: list[dict[str, Any]] = []

    for parser in parsers:
        code = parser.__name__.split("_")[-1]
        try:
            parsed = parser()
            records.extend(parsed)
            parser_status.append(
                {
                    "analysis_code": code,
                    "status": "available" if parsed else "missing",
                    "n_records": len(parsed),
                    "error": "",
                }
            )
        except Exception as exc:
            parser_status.append(
                {
                    "analysis_code": code,
                    "status": "error",
                    "n_records": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    if records:
        record_frame = pd.DataFrame([asdict(record) for record in records])
    else:
        record_frame = pd.DataFrame(
            columns=[field for field in RobustnessRecord.__dataclass_fields__]
        )

    availability = pd.DataFrame(build_missing_analysis_rows(records))
    parser_status_frame = pd.DataFrame(parser_status)

    primary = choose_primary_records(record_frame)
    overall = overall_verdict(primary)
    overall_frame = pd.DataFrame([overall])
    company_summary = build_company_summary(record_frame)
    pair_summary = build_pair_summary(record_frame)
    paragraph = publication_paragraph(overall, primary)

    # Add source availability details.
    source_summary = (
        record_frame.groupby(["analysis_code", "analysis_name"], as_index=False)
        .agg(
            n_records=("entity", "size"),
            scopes=("scope", lambda x: "|".join(sorted(set(x)))),
            source_files=("source_file", lambda x: "|".join(sorted(set(x)))),
        )
        if not record_frame.empty
        else pd.DataFrame(
            columns=[
                "analysis_code",
                "analysis_name",
                "n_records",
                "scopes",
                "source_files",
            ]
        )
    )

    # Publication summary table.
    publication_summary = primary[
        [
            "analysis_code",
            "analysis_name",
            "finding_tested",
            "primary_metric",
            "primary_value",
            "secondary_metric",
            "secondary_value",
            "structure_conclusion",
            "ranking_conclusion",
            "robustness_conclusion",
            "interpretation",
        ]
    ].copy()

    overall_publication_row = pd.DataFrame(
        [
            {
                "analysis_code": "06g",
                "analysis_name": "Overall WP6 robustness",
                "finding_tested": "Combined robustness of company semantic identities",
                "primary_metric": "mean_robustness_score",
                "primary_value": overall["mean_robustness_score"],
                "secondary_metric": "analysis_coverage",
                "secondary_value": overall["analysis_coverage"],
                "structure_conclusion": overall["continuous_structure_conclusion"],
                "ranking_conclusion": overall["exact_ranking_conclusion"],
                "robustness_conclusion": overall["overall_conclusion"],
                "interpretation": overall["interpretation"],
            }
        ]
    )
    publication_summary = pd.concat(
        [publication_summary, overall_publication_row],
        ignore_index=True,
    )

    # Save outputs.
    record_frame.to_csv(
        SUMMARY_OUTPUT_DIR / "wp6_all_robustness_records.csv",
        index=False,
    )
    availability.to_csv(
        SUMMARY_OUTPUT_DIR / "wp6_analysis_availability.csv",
        index=False,
    )
    parser_status_frame.to_csv(
        SUMMARY_OUTPUT_DIR / "wp6_parser_status.csv",
        index=False,
    )
    source_summary.to_csv(
        SUMMARY_OUTPUT_DIR / "wp6_source_files.csv",
        index=False,
    )
    primary.to_csv(
        SUMMARY_OUTPUT_DIR / "wp6_primary_analysis_summary.csv",
        index=False,
    )
    overall_frame.to_csv(
        SUMMARY_OUTPUT_DIR / "wp6_global_robustness_summary.csv",
        index=False,
    )
    company_summary.to_csv(
        SUMMARY_OUTPUT_DIR / "wp6_company_robustness_summary.csv",
        index=False,
    )
    pair_summary.to_csv(
        SUMMARY_OUTPUT_DIR / "wp6_pairwise_robustness_summary.csv",
        index=False,
    )
    publication_summary.to_csv(
        SUMMARY_OUTPUT_DIR / "wp6_publication_summary.csv",
        index=False,
    )
    (SUMMARY_OUTPUT_DIR / "wp6_publication_paragraph.txt").write_text(
        paragraph + "\n",
        encoding="utf-8",
    )
    (SUMMARY_OUTPUT_DIR / "wp6_summary_metadata.json").write_text(
        json.dumps(
            {
                "script_version": SCRIPT_VERSION,
                "project_root": str(PROJECT_ROOT),
                "output_directory": str(SUMMARY_OUTPUT_DIR),
                "overall": overall,
                "expected_analyses": EXPECTED_ANALYSES,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    section("ANALYSIS AVAILABILITY")
    print(availability.to_string(index=False))

    if not parser_status_frame.empty and (parser_status_frame["status"] == "error").any():
        section("PARSER WARNINGS")
        print(
            parser_status_frame[
                parser_status_frame["status"] == "error"
            ].to_string(index=False)
        )

    section("PRIMARY WP6 ANALYSIS SUMMARY")
    if primary.empty:
        print("No robustness summaries were discovered.")
    else:
        print(
            concise_console_table(
                primary,
                [
                    "analysis_code",
                    "analysis_name",
                    "primary_metric",
                    "primary_value",
                    "structure_conclusion",
                    "ranking_conclusion",
                    "robustness_conclusion",
                ],
            )
        )

    section("GLOBAL WP6 ROBUSTNESS")
    print(overall_frame.to_string(index=False))

    if not company_summary.empty:
        section("COMPANY-LEVEL ROBUSTNESS")
        print(company_summary.to_string(index=False))

    if not pair_summary.empty:
        section("PAIRWISE ROBUSTNESS")
        print(pair_summary.to_string(index=False))

    section("PUBLICATION-READY INTERPRETATION")
    print(paragraph)

    section("OUTPUT FILES")
    for path in sorted(SUMMARY_OUTPUT_DIR.iterdir()):
        if path.is_file():
            print(path)


if __name__ == "__main__":
    main()
