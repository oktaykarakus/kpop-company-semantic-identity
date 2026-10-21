"""
09a_cross_workpackage_consistency.py

WP9a — Cross-Work-Package Consistency
Project: Semantic Fingerprints of K-pop Entertainment Companies

Scientific question
-------------------
Do WP3-WP8 tell a consistent organisational story?

Principles
----------
- WP1-WP8 are treated as frozen.
- No new embeddings, graphs, clustering, prediction models, latent variables,
  figures, or composite indices are created.
- Existing outputs are reused only.
- Conflicting, qualifying, neutral, and unavailable evidence are retained.
- Exact known paths are used for WP3-WP6.
- WP7-WP8 files are resolved by exact filename only; folder names are not guessed.
  Resolution fails if an exact filename is absent or appears more than once.

Outputs
-------
outputs/framework_validation/09a3_cross_workpackage_consistency/
    cross_wp_evidence_ledger.csv
    cross_wp_consistency_summary.csv
    cross_wp_conflicts.csv
    cross_wp_input_manifest.csv
    cross_wp_run_summary.csv

Run
---
python 09a4_cross_workpackage_consistency.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Iterable

import numpy as np
import pandas as pd


# =============================================================================
# Project paths
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "framework_validation"
    / "09a4_cross_workpackage_consistency"
)

COMPANIES = ("HYBE", "JYP", "SM", "YG")

# Exact, verified WP3-WP6 paths from the project inventory.
KNOWN_INPUTS = {
    "WP3_company_profiles": (
        ROOT
        / "outputs"
        / "company_profiles"
        / "tables"
        / "table_company_semantic_profiles.csv"
    ),
    "WP3_company_diversity": (
        ROOT
        / "outputs"
        / "company_profiles"
        / "tables"
        / "company_semantic_diversity.csv"
    ),
    "WP3_pairwise_similarity": (
        ROOT
        / "outputs"
        / "company_profiles"
        / "tables"
        / "pairwise_company_profile_similarity.csv"
    ),
    "WP4_global_tests": (
        ROOT
        / "outputs"
        / "company_significance"
        / "tables"
        / "artist_level_global_metric_tests.csv"
    ),
    "WP4_pairwise_tests": (
        ROOT
        / "outputs"
        / "company_significance"
        / "tables"
        / "artist_level_pairwise_metric_tests.csv"
    ),
    "WP4_enrichment": (
        ROOT
        / "outputs"
        / "company_significance"
        / "tables"
        / "community_company_enrichment.csv"
    ),
    "WP4_bootstrap": (
        ROOT
        / "outputs"
        / "company_significance"
        / "tables"
        / "company_diversity_bootstrap_summary.csv"
    ),
    "WP5_drift_summary": (
        ROOT
        / "outputs"
        / "temporal"
        / "rolling5"
        / "semantic_drift_summary_consecutive.csv"
    ),
    "WP5_pairwise_drift": (
        ROOT
        / "outputs"
        / "temporal"
        / "rolling5"
        / "semantic_drift_pairwise_consecutive.csv"
    ),
    "WP5_global_drift_test": (
        ROOT
        / "outputs"
        / "temporal"
        / "rolling5"
        / "semantic_drift_kruskal_consecutive.csv"
    ),
    "WP5_change_points": (
        ROOT
        / "outputs"
        / "temporal"
        / "rolling5"
        / "semantic_change_point_summary.csv"
    ),
    "WP5_stability": (
        ROOT
        / "outputs"
        / "temporal"
        / "rolling5"
        / "temporal_stability_summary.csv"
    ),
    "WP5_trajectory": (
        ROOT
        / "outputs"
        / "temporal"
        / "rolling5"
        / "temporal_trajectory_summary.csv"
    ),
    "WP6_all_records": (
        ROOT
        / "outputs"
        / "robustness"
        / "summary_robustness"
        / "wp6_all_robustness_records.csv"
    ),
    "WP6_company_summary": (
        ROOT
        / "outputs"
        / "robustness"
        / "summary_robustness"
        / "wp6_company_robustness_summary.csv"
    ),
    "WP6_global_summary": (
        ROOT
        / "outputs"
        / "robustness"
        / "summary_robustness"
        / "wp6_global_robustness_summary.csv"
    ),
}

# Exact filenames only. No alternative folder list is searched.
EXACT_FILENAME_INPUTS = {
    "WP7_key_results": "wp7_key_results.csv",
    "WP7_company_profile": "wp7_company_profile_summary.csv",
    "WP7_statistical_evidence": "wp7_statistical_evidence.csv",
    "WP8_integrated_identity": "integrated_company_identity.csv",
    "WP8_company_summary": "company_identity_summary.csv",
    "WP8_traceability": "identity_evidence_traceability.csv",
    "WP8_cross_consistency": "cross_analysis_consistency.csv",
}


# =============================================================================
# Claim taxonomy
# =============================================================================

@dataclass(frozen=True)
class ClaimRule:
    category: str
    keywords: tuple[str, ...]
    supportive_metric_tokens: tuple[str, ...]
    inverse_metric_tokens: tuple[str, ...]


CLAIM_RULES = (
    ClaimRule(
        category="novelty_exploration",
        keywords=(
            "novel", "explor", "innovation", "specialist",
            "distinctive", "distinctiveness", "signature",
        ),
        supportive_metric_tokens=(
            "novelty", "distinctiveness", "distance", "signature",
            "bridge",
        ),
        inverse_metric_tokens=("similarity",),
    ),
    ClaimRule(
        category="integration",
        keywords=(
            "integrat", "ecosystem", "cohesive", "structural integration",
        ),
        supportive_metric_tokens=(
            "participation", "neighbour_entropy", "community_entropy",
            "integration", "overlap", "similarity",
        ),
        inverse_metric_tokens=("distance", "novelty"),
    ),
    ClaimRule(
        category="curated_compactness",
        keywords=(
            "curated", "compact", "concentrated",
            "focused portfolio", "curated semantic portfolio",
        ),
        supportive_metric_tokens=(
            "hhi", "concentration", "dominance",
            "dominant_community_share", "top_three_communities_share",
        ),
        inverse_metric_tokens=(
            "entropy", "effective_n_communities", "coverage",
            "simpson", "communities_occupied",
        ),
    ),
    ClaimRule(
        category="breadth_diversity",
        keywords=(
            "breadth", "divers", "generalist", "broad", "heterogeneous",
            "portfolio", "coverage",
        ),
        supportive_metric_tokens=(
            "entropy", "effective_n_communities", "coverage",
            "simpson", "communities_occupied",
        ),
        inverse_metric_tokens=("hhi", "concentration", "dominance"),
    ),
    ClaimRule(
        category="adaptation_dynamics",
        keywords=(
            "adapt", "dynamic", "evolving", "change", "temporal",
            "movement", "drift",
        ),
        supportive_metric_tokens=(
            "mean_js", "median_js", "drift", "change_point",
            "instability", "volatility", "transition",
        ),
        inverse_metric_tokens=("stability_index", "stable_transition_rate"),
    ),
    ClaimRule(
        category="stability",
        keywords=(
            "stable", "stability", "consistent", "balanced",
        ),
        supportive_metric_tokens=(
            "stability_index", "stable_transition_rate", "robustness",
            "preservation", "retention",
        ),
        inverse_metric_tokens=(
            "mean_js", "median_js", "drift", "change_point",
            "instability", "volatility",
        ),
    ),
    ClaimRule(
        category="predictive_identifiability",
        keywords=(
            "predict", "identif", "classif", "forecast", "persistence",
        ),
        supportive_metric_tokens=(
            "accuracy", "balanced_accuracy", "f1", "auc",
            "identifiability", "forecast", "permutation",
        ),
        inverse_metric_tokens=("error", "rmse", "mae"),
    ),
    ClaimRule(
        category="robustness",
        keywords=(
            "robust", "sensitive", "reproduc", "retained", "preserved",
        ),
        supportive_metric_tokens=(
            "robustness", "probability", "retention", "preservation",
            "agreement", "correlation",
        ),
        inverse_metric_tokens=("maximum_change", "rank_displacement", "bias"),
    ),
)


# =============================================================================
# Utility functions
# =============================================================================

def normalise_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def safe_numeric(value: object) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return np.nan
    return converted if np.isfinite(converted) else np.nan


def canonical_company(value: object) -> str:
    text = str(value).strip().upper()
    return text if text in COMPANIES else ""


def read_csv_required(path: Path, required_columns: Iterable[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input does not exist: {path}")

    data = pd.read_csv(path)
    missing = sorted(set(required_columns) - set(data.columns))

    if missing:
        raise ValueError(
            f"{path.relative_to(ROOT)} is missing required columns: {missing}"
        )

    return data


def resolve_exact_filename(filename: str) -> Path:
    """
    Resolve one exact filename within ROOT/outputs.

    This does not try alternative directories or guessed paths. It searches for
    one exact basename and requires exactly one match.
    """
    output_root = ROOT / "outputs"

    if not output_root.exists():
        raise FileNotFoundError(f"Output root does not exist: {output_root}")

    matches = sorted(
        path
        for path in output_root.rglob(filename)
        if path.is_file() and path.name == filename
    )

    if not matches:
        raise FileNotFoundError(
            f"Exact required filename was not found under outputs/: {filename}"
        )

    if len(matches) > 1:
        relative = [str(path.relative_to(ROOT)) for path in matches]
        raise RuntimeError(
            f"Exact filename is ambiguous ({filename}). Matches: {relative}"
        )

    return matches[0]


def classify_claim_category(text: object) -> str:
    normalised = normalise_text(text)

    curated_phrases = (
        "curated semantic portfolio",
        "curated portfolio",
        "compact semantic portfolio",
        "concentrated semantic portfolio",
        "focused semantic portfolio",
    )
    if any(phrase in normalised for phrase in curated_phrases):
        return "curated_compactness"

    scores = {
        rule.category: sum(keyword in normalised for keyword in rule.keywords)
        for rule in CLAIM_RULES
    }

    best_category = max(scores, key=scores.get)
    return best_category if scores[best_category] > 0 else "other"


def is_metric_relevant(
    metric_name: str,
    claim_category: str,
    claim_text: str = "",
) -> bool:
    """Return whether a frozen metric is scientifically relevant to a claim."""
    metric = normalise_text(metric_name)
    claim = normalise_text(claim_text)

    allowed_tokens = {
        "novelty_exploration": (
            "novelty", "distinctiveness", "distance", "signature",
        ),
        "integration": (
            "participation", "neighbour_entropy", "community_entropy",
            "external_strength", "integration", "overlap", "similarity",
        ),
        "breadth_diversity": (
            "entropy", "effective_n_communities", "coverage",
            "simpson", "hhi", "concentration", "dominance",
        ),
        "curated_compactness": (
            "hhi", "concentration", "dominance",
            "dominant_community_share", "top_three_communities_share",
            "entropy", "effective_n_communities", "coverage", "simpson",
        ),
        "adaptation_dynamics": (
            "mean_js", "median_js", "drift", "change_point",
            "instability", "volatility", "transition", "trajectory",
        ),
        "stability": (
            "stability", "stable_transition", "mean_js", "median_js",
            "drift", "change_point", "instability", "volatility",
        ),
        "predictive_identifiability": (
            "accuracy", "balanced_accuracy", "f1", "auc",
            "identifiability", "forecast", "permutation",
            "error", "rmse", "mae",
        ),
        "robustness": (
            "robustness", "retention", "preservation", "agreement",
            "correlation", "rank", "bias",
        ),
    }

    # Distinctiveness-only claims are not automatically supported or opposed by
    # song-level novelty or participation unless novelty/exploration is explicit.
    pure_distinctiveness = (
        any(token in claim for token in ("distinctiveness", "signature strength"))
        and not any(token in claim for token in ("novel", "explor"))
    )
    if pure_distinctiveness and any(
        token in metric
        for token in ("centroid_novelty", "local_novelty", "participation")
    ):
        return False

    return any(
        token in metric
        for token in allowed_tokens.get(claim_category, ())
    )


def metric_direction(metric_name: str, category: str) -> int:
    """
    Return +1 if larger values usually support the category,
    -1 if smaller values usually support it, and 0 if undetermined.
    """
    metric = normalise_text(metric_name)

    for rule in CLAIM_RULES:
        if rule.category != category:
            continue

        if any(token in metric for token in rule.inverse_metric_tokens):
            return -1

        if any(token in metric for token in rule.supportive_metric_tokens):
            return 1

    return 0


def percentile_position(
    values: pd.Series,
    target_value: float,
    higher_is_supportive: bool,
) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()

    if numeric.empty or not np.isfinite(target_value):
        return np.nan

    if len(numeric) == 1:
        return 0.5

    less_or_equal = float((numeric <= target_value).mean())
    position = less_or_equal if higher_is_supportive else 1.0 - less_or_equal
    return position


def evidence_status_from_percentile(position: float) -> str:
    if pd.isna(position):
        return "unavailable"
    if position >= 0.75:
        return "supporting"
    if position <= 0.25:
        return "conflicting"
    return "qualifying"


def text_robustness_status(text: object) -> str:
    value = normalise_text(text)

    negative = (
        "not robust",
        "structure lost",
        "finding not retained",
        "not retained",
        "not preserved",
        "failed robustness",
    )
    qualifying = (
        "sensitive",
        "rank_sensitive",
        "rank sensitive",
        "sampling_sensitive",
        "sampling sensitive",
        "ordering changed",
        "ranking changed",
        "minor local changes",
        "mixed",
    )
    positive = (
        "highly_robust",
        "highly robust",
        "robust",
        "stable",
        "retained",
        "preserved",
        "consistent",
    )

    if any(token in value for token in negative):
        return "conflicting"
    if any(token in value for token in qualifying):
        return "qualifying"
    if any(token in value for token in positive):
        return "supporting"
    return "qualifying" if value else "unavailable"


def text_significance_status(
    significant: object,
    direction_matches: bool | None = None,
) -> str:
    is_significant = bool(significant)

    if not is_significant:
        return "qualifying"

    if direction_matches is False:
        return "conflicting"

    return "supporting"


def make_evidence_row(
    *,
    company: str,
    work_package: str,
    source_file: Path,
    source_row: str,
    analysis_component: str,
    claim_category: str,
    evidence_statement: str,
    metric: str = "",
    value: object = np.nan,
    comparison: str = "",
    evidence_status: str,
    direction: str = "",
    statistical_status: str = "",
    rationale: str = "",
    originating_claim: str = "",
    evidence_relevance: str = "Primary",
) -> dict[str, object]:
    return {
        "company": company,
        "work_package": work_package,
        "analysis_component": analysis_component,
        "claim_category": claim_category,
        "originating_claim": originating_claim,
        "evidence_statement": evidence_statement,
        "metric": metric,
        "value": value,
        "comparison": comparison,
        "direction": direction,
        "statistical_status": statistical_status,
        "evidence_status": evidence_status,
        "evidence_relevance": evidence_relevance,
        "rationale": rationale,
        "source_file": str(source_file.relative_to(ROOT)),
        "source_row": source_row,
    }


# =============================================================================
# Claim extraction from WP8
# =============================================================================

def extract_wp8_claims(paths: dict[str, Path]) -> pd.DataFrame:
    traceability = pd.read_csv(paths["WP8_traceability"])

    company_col = next(
        (column for column in ("company", "entity") if column in traceability.columns),
        None,
    )
    claim_col = next(
        (
            column
            for column in (
                "claim",
                "conclusion",
                "interpretation",
                "integrated_identity",
                "identity",
            )
            if column in traceability.columns
        ),
        None,
    )

    if company_col is None or claim_col is None:
        raise ValueError(
            "identity_evidence_traceability.csv must include a company/entity "
            "column and a claim/conclusion/interpretation column."
        )

    claims = traceability[[company_col, claim_col]].copy()
    claims.columns = ["company", "claim"]
    claims["company"] = claims["company"].map(canonical_company)
    claims["claim"] = claims["claim"].astype(str).str.strip()
    claims = claims[
        claims["company"].isin(COMPANIES)
        & claims["claim"].ne("")
        & claims["claim"].ne("nan")
    ].drop_duplicates()

    # Ensure final integrated identity is represented even if traceability omits it.
    identity = pd.read_csv(paths["WP8_integrated_identity"])
    identity_company_col = next(
        (column for column in ("company", "entity") if column in identity.columns),
        None,
    )
    identity_claim_col = next(
        (
            column
            for column in (
                "integrated_identity",
                "organisational_identity",
                "identity",
                "final_identity",
                "interpretation",
            )
            if column in identity.columns
        ),
        None,
    )

    if identity_company_col and identity_claim_col:
        identity_claims = identity[
            [identity_company_col, identity_claim_col]
        ].copy()
        identity_claims.columns = ["company", "claim"]
        identity_claims["company"] = identity_claims["company"].map(
            canonical_company
        )
        identity_claims["claim"] = identity_claims["claim"].astype(str).str.strip()
        identity_claims = identity_claims[
            identity_claims["company"].isin(COMPANIES)
            & identity_claims["claim"].ne("")
            & identity_claims["claim"].ne("nan")
        ]
        claims = pd.concat([claims, identity_claims], ignore_index=True).drop_duplicates()

    claims["claim_id"] = [
        f"{company}_C{index:03d}"
        for index, company in enumerate(claims["company"], start=1)
    ]
    claims["claim_category"] = claims["claim"].map(classify_claim_category)

    return claims[
        ["claim_id", "company", "claim_category", "claim"]
    ].sort_values(["company", "claim_id"])


# =============================================================================
# WP3 evidence
# =============================================================================

def parse_wp3(
    paths: dict[str, Path],
    claims: pd.DataFrame,
) -> list[dict[str, object]]:
    evidence: list[dict[str, object]] = []

    profile = read_csv_required(
        paths["WP3_company_profiles"],
        ["company"],
    )
    diversity = read_csv_required(
        paths["WP3_company_diversity"],
        ["company"],
    )

    profile["company"] = profile["company"].map(canonical_company)
    diversity["company"] = diversity["company"].map(canonical_company)

    company_tables = (
        ("Company semantic profile", paths["WP3_company_profiles"], profile),
        ("Company semantic diversity", paths["WP3_company_diversity"], diversity),
    )

    numeric_candidates = {
        "novelty_exploration": (
            "mean_centroid_novelty",
            "mean_local_novelty",
            "mean_participation",
        ),
        "integration": (
            "mean_participation",
            "mean_neighbour_entropy",
        ),
        "breadth_diversity": (
            "normalised_shannon_entropy",
            "effective_n_communities",
            "community_coverage_fraction",
            "simpson_diversity",
            "hhi_concentration",
        ),
        "curated_compactness": (
            "hhi_concentration",
            "gini_concentration",
            "dominant_community_share",
            "top_three_communities_share",
            "normalised_shannon_entropy",
            "effective_n_communities",
        ),
    }

    for _, claim in claims.iterrows():
        category = claim["claim_category"]

        if category not in numeric_candidates:
            continue

        for analysis_component, source_path, table in company_tables:
            company_row = table[table["company"] == claim["company"]]

            if company_row.empty:
                continue

            for metric in numeric_candidates[category]:
                if metric not in table.columns:
                    continue

                if not is_metric_relevant(
                    metric_name=metric,
                    claim_category=category,
                    claim_text=claim["claim"],
                ):
                    continue

                value = safe_numeric(company_row.iloc[0][metric])
                direction = metric_direction(metric, category)

                if direction == 0 or not np.isfinite(value):
                    status = "qualifying"
                    position = np.nan
                else:
                    position = percentile_position(
                        table[metric],
                        value,
                        higher_is_supportive=direction > 0,
                    )
                    status = evidence_status_from_percentile(position)

                rationale = (
                    f"Company-relative percentile position={position:.3f}; "
                    f"{'higher' if direction > 0 else 'lower' if direction < 0 else 'no fixed direction'} "
                    f"values were treated as more compatible with {category}."
                    if np.isfinite(position)
                    else "The metric was retained descriptively because a directional "
                    "claim mapping was unavailable."
                )

                evidence.append(
                    make_evidence_row(
                        company=claim["company"],
                        work_package="WP3",
                        source_file=source_path,
                        source_row=f"company={claim['company']}",
                        analysis_component=analysis_component,
                        claim_category=category,
                        originating_claim=claim["claim"],
                        evidence_statement=(
                            f"{claim['company']} has {metric}={value:.6g}."
                        ),
                        metric=metric,
                        value=value,
                        comparison="Relative position among four companies",
                        evidence_status=status,
                        direction=(
                            "higher_supports"
                            if direction > 0
                            else "lower_supports"
                            if direction < 0
                            else "descriptive"
                        ),
                        rationale=rationale,
                    )
                )

    pairwise = read_csv_required(
        paths["WP3_pairwise_similarity"],
        ["company_1", "company_2"],
    )

    for _, row in pairwise.iterrows():
        company_1 = canonical_company(row["company_1"])
        company_2 = canonical_company(row["company_2"])

        for company in (company_1, company_2):
            if not company:
                continue

            company_claims = claims[
                (claims["company"] == company)
                & claims["claim_category"].isin(
                    ["novelty_exploration", "integration"]
                )
            ]

            for _, claim in company_claims.iterrows():
                for metric in (
                    "cosine_similarity",
                    "jensen_shannon_distance",
                    "profile_overlap",
                ):
                    if metric not in pairwise.columns:
                        continue

                    value = safe_numeric(row[metric])
                    category = claim["claim_category"]
                    direction = metric_direction(metric, category)

                    evidence.append(
                        make_evidence_row(
                            company=company,
                            work_package="WP3",
                            source_file=paths["WP3_pairwise_similarity"],
                            source_row=(
                                f"company_1={company_1};company_2={company_2}"
                            ),
                            analysis_component="Pairwise company profile geometry",
                            claim_category=category,
                            originating_claim=claim["claim"],
                            evidence_statement=(
                                f"{company_1}-{company_2} {metric}={value:.6g}."
                            ),
                            metric=metric,
                            value=value,
                            comparison=f"{company_1} vs {company_2}",
                            evidence_status="qualifying",
                            direction=(
                                "higher_supports"
                                if direction > 0
                                else "lower_supports"
                                if direction < 0
                                else "descriptive"
                            ),
                            rationale=(
                                "Pairwise geometry is retained as qualifying evidence; "
                                "it is not converted into a company-level support judgment "
                                "without a separate frozen decision rule."
                            ),
                        )
                    )

    return evidence


# =============================================================================
# WP4 evidence
# =============================================================================

def parse_wp4(
    paths: dict[str, Path],
    claims: pd.DataFrame,
) -> list[dict[str, object]]:
    evidence: list[dict[str, object]] = []

    global_tests = read_csv_required(
        paths["WP4_global_tests"],
        ["metric", "holm_adjusted_p", "significant_holm_005"],
    )

    for _, row in global_tests.iterrows():
        metric = str(row["metric"])
        p_value = safe_numeric(row["holm_adjusted_p"])
        significant = bool(row["significant_holm_005"])

        for _, claim in claims.iterrows():
            if not is_metric_relevant(
                metric_name=metric,
                claim_category=claim["claim_category"],
                claim_text=claim["claim"],
            ):
                continue

            direction = metric_direction(metric, claim["claim_category"])

            if direction == 0:
                continue

            evidence.append(
                make_evidence_row(
                    company=claim["company"],
                    work_package="WP4",
                    source_file=paths["WP4_global_tests"],
                    source_row=f"metric={metric}",
                    analysis_component="Artist-level global metric test",
                    claim_category=claim["claim_category"],
                    originating_claim=claim["claim"],
                    evidence_statement=(
                        f"Global company difference for {metric}: "
                        f"Holm-adjusted p={p_value:.6g}, "
                        f"significant={significant}."
                    ),
                    metric=metric,
                    value=p_value,
                    comparison="Four-company global test",
                    evidence_status=(
                        "supporting" if significant else "qualifying"
                    ),
                    statistical_status=(
                        "significant_holm_005"
                        if significant
                        else "not_significant_holm_005"
                    ),
                    rationale=(
                        "A significant global test supports the existence of "
                        "company differentiation for this construct but does not "
                        "by itself establish a company-specific direction."
                    ),
                )
            )

    pairwise = read_csv_required(
        paths["WP4_pairwise_tests"],
        [
            "metric",
            "company_1",
            "company_2",
            "mean_difference_1_minus_2",
            "holm_adjusted_p",
            "significant_holm_005",
        ],
    )

    for _, row in pairwise.iterrows():
        metric = str(row["metric"])
        company_1 = canonical_company(row["company_1"])
        company_2 = canonical_company(row["company_2"])
        difference = safe_numeric(row["mean_difference_1_minus_2"])
        p_value = safe_numeric(row["holm_adjusted_p"])
        significant = bool(row["significant_holm_005"])

        for company, sign in ((company_1, 1.0), (company_2, -1.0)):
            if not company:
                continue

            company_claims = claims[claims["company"] == company]

            for _, claim in company_claims.iterrows():
                if not is_metric_relevant(
                    metric_name=metric,
                    claim_category=claim["claim_category"],
                    claim_text=claim["claim"],
                ):
                    continue

                direction = metric_direction(metric, claim["claim_category"])

                if direction == 0:
                    continue

                company_advantage = sign * difference
                direction_matches = (
                    company_advantage > 0 if direction > 0 else company_advantage < 0
                )

                evidence.append(
                    make_evidence_row(
                        company=company,
                        work_package="WP4",
                        source_file=paths["WP4_pairwise_tests"],
                        source_row=(
                            f"metric={metric};company_1={company_1};"
                            f"company_2={company_2}"
                        ),
                        analysis_component="Artist-level pairwise metric test",
                        claim_category=claim["claim_category"],
                        originating_claim=claim["claim"],
                        evidence_statement=(
                            f"{company_1}-{company_2} mean difference for "
                            f"{metric}={difference:.6g}; Holm-adjusted "
                            f"p={p_value:.6g}."
                        ),
                        metric=metric,
                        value=company_advantage,
                        comparison=f"{company_1} vs {company_2}",
                        evidence_status=text_significance_status(
                            significant,
                            direction_matches=direction_matches,
                        ),
                        direction=(
                            "claim_compatible"
                            if direction_matches
                            else "claim_incompatible"
                        ),
                        statistical_status=(
                            "significant_holm_005"
                            if significant
                            else "not_significant_holm_005"
                        ),
                        rationale=(
                            "Direction was evaluated against the claim taxonomy; "
                            "nonsignificant contrasts are retained as qualifying "
                            "rather than supportive evidence."
                        ),
                    )
                )

    bootstrap = read_csv_required(
        paths["WP4_bootstrap"],
        ["company", "metric", "point_estimate", "ci_low", "ci_high"],
    )
    bootstrap["company"] = bootstrap["company"].map(canonical_company)

    for _, row in bootstrap.iterrows():
        company = row["company"]
        metric = str(row["metric"])

        if company not in COMPANIES:
            continue

        company_claims = claims[claims["company"] == company]

        for _, claim in company_claims.iterrows():
            if not is_metric_relevant(
                metric_name=metric,
                claim_category=claim["claim_category"],
                claim_text=claim["claim"],
            ):
                continue

            direction = metric_direction(metric, claim["claim_category"])

            if direction == 0:
                continue

            evidence.append(
                make_evidence_row(
                    company=company,
                    work_package="WP4",
                    source_file=paths["WP4_bootstrap"],
                    source_row=f"company={company};metric={metric}",
                    analysis_component="Artist-block bootstrap uncertainty",
                    claim_category=claim["claim_category"],
                    originating_claim=claim["claim"],
                    evidence_statement=(
                        f"{metric}={safe_numeric(row['point_estimate']):.6g}, "
                        f"95% bootstrap interval "
                        f"[{safe_numeric(row['ci_low']):.6g}, "
                        f"{safe_numeric(row['ci_high']):.6g}]."
                    ),
                    metric=metric,
                    value=safe_numeric(row["point_estimate"]),
                    comparison="Within-company bootstrap uncertainty",
                    evidence_status="qualifying",
                    direction="uncertainty_estimate",
                    rationale=(
                        "Bootstrap intervals quantify uncertainty but are not "
                        "converted into support/conflict without a frozen "
                        "cross-company decision threshold."
                    ),
                )
            )

    return evidence


# =============================================================================
# WP5 evidence
# =============================================================================

def parse_wp5(
    paths: dict[str, Path],
    claims: pd.DataFrame,
) -> list[dict[str, object]]:
    evidence: list[dict[str, object]] = []

    drift = read_csv_required(
        paths["WP5_drift_summary"],
        ["company", "mean_js", "median_js", "n_transitions"],
    )
    stability = read_csv_required(
        paths["WP5_stability"],
        [
            "company",
            "stability_index",
            "stability_rank",
            "stability_category",
            "change_point_rate",
        ],
    )
    trajectory = read_csv_required(
        paths["WP5_trajectory"],
        ["company", "trend_classification", "ols_p_value", "kendall_p_value"],
    )

    for table in (drift, stability, trajectory):
        table["company"] = table["company"].map(canonical_company)

    metric_map = {
        "adaptation_dynamics": (
            (drift, paths["WP5_drift_summary"], "mean_js"),
            (drift, paths["WP5_drift_summary"], "median_js"),
            (stability, paths["WP5_stability"], "change_point_rate"),
            (
                stability,
                paths["WP5_stability"],
                "relative_instability_score",
            ),
        ),
        "stability": (
            (stability, paths["WP5_stability"], "stability_index"),
            (
                stability,
                paths["WP5_stability"],
                "stable_transition_rate",
            ),
            (drift, paths["WP5_drift_summary"], "mean_js"),
        ),
    }

    for _, claim in claims.iterrows():
        category = claim["claim_category"]

        if category not in metric_map:
            continue

        for table, source_path, metric in metric_map[category]:
            if metric not in table.columns:
                continue

            company_row = table[table["company"] == claim["company"]]

            if company_row.empty:
                continue

            value = safe_numeric(company_row.iloc[0][metric])
            direction = metric_direction(metric, category)
            position = percentile_position(
                table[metric],
                value,
                higher_is_supportive=direction > 0,
            )
            status = evidence_status_from_percentile(position)

            evidence.append(
                make_evidence_row(
                    company=claim["company"],
                    work_package="WP5",
                    source_file=source_path,
                    source_row=f"company={claim['company']}",
                    analysis_component="Rolling five-year temporal dynamics",
                    claim_category=category,
                    originating_claim=claim["claim"],
                    evidence_statement=(
                        f"{claim['company']} has {metric}={value:.6g}."
                    ),
                    metric=metric,
                    value=value,
                    comparison="Relative position among four companies",
                    evidence_status=status,
                    direction=(
                        "higher_supports"
                        if direction > 0
                        else "lower_supports"
                    ),
                    rationale=(
                        f"Company-relative percentile position={position:.3f} "
                        "using consecutive rolling five-year transitions."
                    ),
                )
            )

        company_trajectory = trajectory[
            trajectory["company"] == claim["company"]
        ]

        if not company_trajectory.empty:
            row = company_trajectory.iloc[0]
            evidence.append(
                make_evidence_row(
                    company=claim["company"],
                    work_package="WP5",
                    source_file=paths["WP5_trajectory"],
                    source_row=f"company={claim['company']}",
                    analysis_component="Temporal trajectory trend",
                    claim_category=category,
                    originating_claim=claim["claim"],
                    evidence_statement=(
                        f"Trend classification={row['trend_classification']}; "
                        f"OLS p={safe_numeric(row['ols_p_value']):.6g}; "
                        f"Kendall p={safe_numeric(row['kendall_p_value']):.6g}."
                    ),
                    metric="trend_classification",
                    value=row["trend_classification"],
                    comparison="Within-company temporal trajectory",
                    evidence_status="qualifying",
                    statistical_status=(
                        "trajectory_significance_reported"
                    ),
                    rationale=(
                        "Trend direction and significance are retained as "
                        "qualifying evidence because a decreasing drift trend "
                        "can coexist with historically high movement."
                    ),
                )
            )

    global_test = read_csv_required(
        paths["WP5_global_drift_test"],
        ["p_value", "significant"],
    ).iloc[0]

    for _, claim in claims[
        claims["claim_category"].isin(["adaptation_dynamics", "stability"])
    ].iterrows():
        significant = bool(global_test["significant"])
        p_value = safe_numeric(global_test["p_value"])

        evidence.append(
            make_evidence_row(
                company=claim["company"],
                work_package="WP5",
                source_file=paths["WP5_global_drift_test"],
                source_row="global_test",
                analysis_component="Global company temporal-drift test",
                claim_category=claim["claim_category"],
                originating_claim=claim["claim"],
                evidence_statement=(
                    f"Company drift differences: p={p_value:.6g}, "
                    f"significant={significant}."
                ),
                metric="global_temporal_drift_p_value",
                value=p_value,
                comparison="Four-company Kruskal-Wallis test",
                evidence_status=(
                    "supporting" if significant else "qualifying"
                ),
                statistical_status=(
                    "significant" if significant else "not_significant"
                ),
                rationale=(
                    "A significant global test supports differentiated temporal "
                    "behaviour but does not alone establish company-specific direction."
                ),
            )
        )

    return evidence


# =============================================================================
# WP6 evidence
# =============================================================================

def parse_wp6(
    paths: dict[str, Path],
    claims: pd.DataFrame,
) -> list[dict[str, object]]:
    evidence: list[dict[str, object]] = []

    records = read_csv_required(
        paths["WP6_all_records"],
        [
            "analysis_code",
            "analysis_name",
            "scope",
            "entity",
            "finding_tested",
            "robustness_conclusion",
            "interpretation",
            "source_file",
        ],
    )

    for index, row in records.iterrows():
        entity = canonical_company(row["entity"])
        text = " ".join(
            [
                str(row.get("finding_tested", "")),
                str(row.get("interpretation", "")),
                str(row.get("analysis_name", "")),
            ]
        )

        category = classify_claim_category(text)
        status = text_robustness_status(
            " ".join(
                [
                    str(row.get("robustness_conclusion", "")),
                    str(row.get("structure_conclusion", "")),
                    str(row.get("ranking_conclusion", "")),
                ]
            )
        )

        target_companies = (
            [entity]
            if entity in COMPANIES
            else list(COMPANIES)
            if normalise_text(row["scope"]) in {"global", "all companies"}
            or "all companies" in normalise_text(row["entity"])
            else []
        )

        for company in target_companies:
            company_claims = claims[claims["company"] == company]

            if category != "other":
                company_claims = company_claims[
                    company_claims["claim_category"] == category
                ]

            if company_claims.empty:
                continue

            for _, claim in company_claims.iterrows():
                evidence.append(
                    make_evidence_row(
                        company=company,
                        work_package="WP6",
                        source_file=paths["WP6_all_records"],
                        source_row=str(index),
                        analysis_component=str(row["analysis_name"]),
                        claim_category=claim["claim_category"],
                        originating_claim=claim["claim"],
                        evidence_statement=str(row["finding_tested"]),
                        metric=str(row.get("primary_metric", "")),
                        value=row.get("primary_value", np.nan),
                        comparison=str(row.get("secondary_metric", "")),
                        evidence_status=status,
                        statistical_status=str(
                            row.get("robustness_conclusion", "")
                        ),
                        rationale=str(row.get("interpretation", "")),
                    )
                )

    return evidence


# =============================================================================
# WP7 evidence
# =============================================================================

def parse_wp7(
    paths: dict[str, Path],
    claims: pd.DataFrame,
) -> list[dict[str, object]]:
    """
    Parse final WP7 summaries conservatively.

    WP7 schemas may differ across final versions, so all columns are retained
    textually and only explicit company rows are assigned to a company.
    """
    evidence: list[dict[str, object]] = []

    for key in (
        "WP7_key_results",
        "WP7_company_profile",
        "WP7_statistical_evidence",
    ):
        path = paths[key]
        table = pd.read_csv(path)

        for index, row in table.iterrows():
            row_text = " | ".join(
                f"{column}={row[column]}"
                for column in table.columns
                if not pd.isna(row[column])
            )
            category = classify_claim_category(row_text)

            company = ""
            for candidate_column in ("company", "entity"):
                if candidate_column in table.columns:
                    company = canonical_company(row[candidate_column])
                    if company:
                        break

            target_companies = [company] if company else list(COMPANIES)

            for target_company in target_companies:
                company_claims = claims[
                    claims["company"] == target_company
                ]

                if category != "other":
                    company_claims = company_claims[
                        company_claims["claim_category"] == category
                    ]

                if company_claims.empty:
                    continue

                for _, claim in company_claims.iterrows():
                    status = "qualifying"
                    lowered = normalise_text(row_text)

                    if any(
                        token in lowered
                        for token in (
                            "significant=true",
                            "significant = true",
                            "robust",
                            "identifiable",
                            "above chance",
                        )
                    ):
                        status = "supporting"

                    if any(
                        token in lowered
                        for token in (
                            "significant=false",
                            "significant = false",
                            "below chance",
                            "not identifiable",
                        )
                    ):
                        status = "qualifying"

                    evidence.append(
                        make_evidence_row(
                            company=target_company,
                            work_package="WP7",
                            source_file=path,
                            source_row=str(index),
                            analysis_component=key.replace("WP7_", ""),
                            claim_category=claim["claim_category"],
                            originating_claim=claim["claim"],
                            evidence_statement=row_text,
                            metric="",
                            value=np.nan,
                            comparison=(
                                "Company-specific"
                                if company
                                else "Global WP7 evidence"
                            ),
                            evidence_status=status,
                            rationale=(
                                "Final WP7 summary evidence was retained without "
                                "re-estimating prediction or forecasting models."
                            ),
                        )
                    )

    return evidence


# =============================================================================
# WP8 evidence
# =============================================================================

def parse_wp8(
    paths: dict[str, Path],
    claims: pd.DataFrame,
) -> list[dict[str, object]]:
    evidence: list[dict[str, object]] = []

    for _, claim in claims.iterrows():
        evidence.append(
            make_evidence_row(
                company=claim["company"],
                work_package="WP8",
                source_file=paths["WP8_traceability"],
                source_row=claim["claim_id"],
                analysis_component="Frozen organisational interpretation",
                claim_category=claim["claim_category"],
                originating_claim=claim["claim"],
                evidence_statement=claim["claim"],
                metric="",
                value=np.nan,
                comparison="Final frozen interpretation",
                evidence_status="supporting",
                rationale=(
                    "This row records the frozen WP8 conclusion being evaluated; "
                    "it is not counted as independent empirical confirmation."
                ),
            )
        )

    consistency = pd.read_csv(paths["WP8_cross_consistency"])

    for index, row in consistency.iterrows():
        row_text = " | ".join(
            f"{column}={row[column]}"
            for column in consistency.columns
            if not pd.isna(row[column])
        )

        company = ""
        for candidate_column in ("company", "entity"):
            if candidate_column in consistency.columns:
                company = canonical_company(row[candidate_column])
                if company:
                    break

        if not company:
            continue

        category = classify_claim_category(row_text)
        company_claims = claims[claims["company"] == company]

        if category != "other":
            company_claims = company_claims[
                company_claims["claim_category"] == category
            ]

        for _, claim in company_claims.iterrows():
            lowered = normalise_text(row_text)
            status = (
                "conflicting"
                if "conflict" in lowered or "inconsistent" in lowered
                else "supporting"
                if "consistent" in lowered or "agreement" in lowered
                else "qualifying"
            )

            evidence.append(
                make_evidence_row(
                    company=company,
                    work_package="WP8",
                    source_file=paths["WP8_cross_consistency"],
                    source_row=str(index),
                    analysis_component="WP8 internal cross-analysis consistency",
                    claim_category=claim["claim_category"],
                    originating_claim=claim["claim"],
                    evidence_statement=row_text,
                    evidence_status=status,
                    rationale=(
                        "Existing WP8 cross-analysis consistency evidence was "
                        "retained without recalculation."
                    ),
                )
            )

    return evidence


# =============================================================================
# Summaries
# =============================================================================

def deduplicate_semantic_evidence(ledger: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated evaluations of the same underlying evidence.

    Raw source-level records remain in the published evidence ledger. This
    deduplicated view is used only for claim summaries and the conflict table,
    preventing the same statistic repeated in two WP summary tables from being
    counted as independent evidence.
    """
    if ledger.empty:
        return ledger.copy()

    data = ledger.copy()
    key_columns = [
        "company",
        "work_package",
        "analysis_component",
        "claim_category",
        "originating_claim",
        "metric",
        "value",
        "comparison",
        "direction",
        "statistical_status",
        "evidence_status",
    ]

    for column in key_columns:
        if column not in data.columns:
            data[column] = ""

    data["_semantic_value"] = data["value"].map(
        lambda value: "" if pd.isna(value) else f"{safe_numeric(value):.12g}"
        if np.isfinite(safe_numeric(value)) else str(value)
    )

    # The WP3 profile and diversity tables can repeat the same frozen company-level
    # statistic. Treat these as one semantic observation, while retaining every
    # contributing file/row/evidence ID in the duplicate traceability columns.
    data["_semantic_component"] = data["analysis_component"].astype(str)
    wp3_company_summary = (
        data["work_package"].eq("WP3")
        & data["analysis_component"].isin(
            ["Company semantic profile", "Company semantic diversity"]
        )
    )
    data.loc[wp3_company_summary, "_semantic_component"] = (
        "WP3 frozen company-level metric"
    )

    effective_keys = [
        "company",
        "work_package",
        "_semantic_component",
        "claim_category",
        "originating_claim",
        "metric",
        "_semantic_value",
        "comparison",
        "direction",
        "statistical_status",
        "evidence_status",
    ]

    rows: list[pd.Series] = []
    for _, group in data.groupby(effective_keys, dropna=False, sort=False):
        representative = group.iloc[0].copy()
        representative["duplicate_record_count"] = int(len(group))
        representative["duplicate_source_files"] = "|".join(
            sorted(set(group["source_file"].dropna().astype(str)))
        )
        representative["duplicate_source_rows"] = "|".join(
            sorted(set(group["source_row"].dropna().astype(str)))
        )
        representative["duplicate_evidence_ids"] = "|".join(
            sorted(set(group["evidence_id"].dropna().astype(str)))
        )
        rows.append(representative)

    result = pd.DataFrame(rows).drop(
        columns=["_semantic_value", "_semantic_component"],
        errors="ignore",
    )
    return result.reset_index(drop=True)


def build_consistency_summary(
    ledger: pd.DataFrame,
    claims: pd.DataFrame,
) -> pd.DataFrame:
    empirical = deduplicate_semantic_evidence(
        ledger[ledger["work_package"] != "WP8"].copy()
    )

    # Unmapped WP8 narrative fragments remain in the evidence ledger for full
    # traceability, but they are excluded from headline consistency statistics.
    mapped_claims = claims[claims["claim_category"] != "other"].copy()

    rows = []

    for _, claim in mapped_claims.iterrows():
        subset = empirical[
            (empirical["company"] == claim["company"])
            & (empirical["originating_claim"] == claim["claim"])
        ]

        work_packages = sorted(subset["work_package"].dropna().unique())
        status_counts = subset["evidence_status"].value_counts()

        n_supporting = int(status_counts.get("supporting", 0))
        n_conflicting = int(status_counts.get("conflicting", 0))
        n_qualifying = int(status_counts.get("qualifying", 0))
        n_unavailable = int(status_counts.get("unavailable", 0))
        n_evidence = int(len(subset))

        available_status = subset[
            subset["evidence_status"].isin(
                ["supporting", "conflicting", "qualifying"]
            )
        ]

        n_available = int(len(available_status))
        agreement_rate = (
            n_supporting / (n_supporting + n_conflicting)
            if (n_supporting + n_conflicting) > 0
            else np.nan
        )
        conflict_rate = (
            n_conflicting / (n_supporting + n_conflicting)
            if (n_supporting + n_conflicting) > 0
            else np.nan
        )
        evidence_coverage = len(work_packages) / 5.0  # WP3-WP7

        # No empirical evidence available.
        if n_available == 0:
            overall = "insufficient_evidence"

        # Genuine disagreement between work packages.
        elif n_conflicting > n_supporting:
            overall = "inconsistent"

        # Some disagreement, but support outweighs conflict.
        elif n_conflicting > 0:
            overall = "mostly_consistent_with_qualifications"

        # Strong support from multiple independent work packages.
        elif n_supporting >= 2:
            overall = "consistent"

        # Some direct support, with the remaining evidence being neutral/qualifying.
        elif n_supporting == 1:
            overall = "consistent_with_qualifications"

        # Only qualifying/contextual evidence available.
        else:
            overall = "insufficient_evidence"

        rows.append(
            {
                "claim_id": claim["claim_id"],
                "company": claim["company"],
                "claim_category": claim["claim_category"],
                "major_conclusion": claim["claim"],
                "n_expected_empirical_work_packages": 5,
                "n_available_work_packages": len(work_packages),
                "available_work_packages": "|".join(work_packages),
                "n_evidence_records": n_evidence,
                "n_supporting": n_supporting,
                "n_conflicting": n_conflicting,
                "n_qualifying": n_qualifying,
                "n_unavailable": n_unavailable,
                "evidence_coverage": evidence_coverage,
                "agreement_rate": agreement_rate,
                "conflict_rate": conflict_rate,
                "overall_consistency": overall,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["company", "claim_category", "claim_id"]
    )


def build_conflicts(ledger: pd.DataFrame) -> pd.DataFrame:
    conflicts = deduplicate_semantic_evidence(
        ledger[ledger["evidence_status"] == "conflicting"].copy()
    )

    if conflicts.empty:
        return pd.DataFrame(
            columns=list(ledger.columns)
            + [
                "duplicate_record_count",
                "duplicate_source_files",
                "duplicate_source_rows",
                "duplicate_evidence_ids",
                "conflict_id",
                "conflict_note",
            ]
        )

    conflicts.insert(
        0,
        "conflict_id",
        [f"CONFLICT_{index:04d}" for index in range(1, len(conflicts) + 1)],
    )
    conflicts["conflict_note"] = (
        "Evidence direction was incompatible with the frozen claim under the "
        "predefined transparent mapping rule. Repeated source-level records of "
        "the same underlying statistic were collapsed for conflict counting."
    )

    return conflicts


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = dict(KNOWN_INPUTS)

    for key, filename in EXACT_FILENAME_INPUTS.items():
        paths[key] = resolve_exact_filename(filename)

    input_manifest_rows = []

    for key, path in paths.items():
        input_manifest_rows.append(
            {
                "input_key": key,
                "work_package": key.split("_", 1)[0],
                "relative_path": str(path.relative_to(ROOT)),
                "exists": path.exists(),
                "resolution_method": (
                    "verified_exact_path"
                    if key in KNOWN_INPUTS
                    else "unique_exact_filename"
                ),
            }
        )

    input_manifest = pd.DataFrame(input_manifest_rows)
    input_manifest.to_csv(
        OUTPUT_DIR / "cross_wp_input_manifest.csv",
        index=False,
    )

    claims = extract_wp8_claims(paths)

    evidence_rows: list[dict[str, object]] = []
    evidence_rows.extend(parse_wp3(paths, claims))
    evidence_rows.extend(parse_wp4(paths, claims))
    evidence_rows.extend(parse_wp5(paths, claims))
    evidence_rows.extend(parse_wp6(paths, claims))
    evidence_rows.extend(parse_wp7(paths, claims))
    evidence_rows.extend(parse_wp8(paths, claims))

    ledger = pd.DataFrame(evidence_rows)

    if ledger.empty:
        raise RuntimeError("No evidence records were produced.")

    ledger.insert(
        0,
        "evidence_id",
        [f"EVID_{index:05d}" for index in range(1, len(ledger) + 1)],
    )

    status_order = pd.CategoricalDtype(
        ["supporting", "conflicting", "qualifying", "unavailable"],
        ordered=True,
    )
    ledger["evidence_status"] = ledger["evidence_status"].astype(status_order)

    ledger = ledger.sort_values(
        [
            "company",
            "originating_claim",
            "work_package",
            "evidence_status",
            "source_file",
            "source_row",
        ]
    ).reset_index(drop=True)

    ledger["evidence_status"] = ledger["evidence_status"].astype(str)

    summary = build_consistency_summary(ledger, claims)
    conflicts = build_conflicts(ledger)

    ledger.to_csv(
        OUTPUT_DIR / "cross_wp_evidence_ledger.csv",
        index=False,
    )
    summary.to_csv(
        OUTPUT_DIR / "cross_wp_consistency_summary.csv",
        index=False,
    )
    conflicts.to_csv(
        OUTPUT_DIR / "cross_wp_conflicts.csv",
        index=False,
    )

    run_summary = pd.DataFrame(
        [
            {
                "analysis": "WP9a4 Cross-Work-Package Consistency",
                "n_companies": claims["company"].nunique(),
                "n_wp8_claims": len(claims),
                "n_mapped_wp8_claims": int((claims["claim_category"] != "other").sum()),
                "n_unmapped_wp8_claims": int((claims["claim_category"] == "other").sum()),
                "n_evidence_records": len(ledger),
                "n_unique_semantic_evidence_records": len(deduplicate_semantic_evidence(ledger)),
                "n_supporting_records": int(
                    (ledger["evidence_status"] == "supporting").sum()
                ),
                "n_conflicting_records": int(
                    (ledger["evidence_status"] == "conflicting").sum()
                ),
                "n_qualifying_records": int(
                    (ledger["evidence_status"] == "qualifying").sum()
                ),
                "n_claims_consistent": int(
                    (summary["overall_consistency"] == "consistent").sum()
                ),
                "n_claims_mostly_consistent": int(
                    (
                        summary["overall_consistency"]
                        == "mostly_consistent_with_qualifications"
                    ).sum()
                ),
                "n_claims_inconsistent": int(
                    (summary["overall_consistency"] == "inconsistent").sum()
                ),
                "output_directory": str(OUTPUT_DIR.relative_to(ROOT)),
            }
        ]
    )
    run_summary.to_csv(
        OUTPUT_DIR / "cross_wp_run_summary.csv",
        index=False,
    )

    print("=" * 88)
    print("WP9a4 CROSS-WORK-PACKAGE CONSISTENCY COMPLETE")
    print("=" * 88)
    print(f"Project root: {ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"WP8 claims extracted: {len(claims):,}")
    print(f"Mapped WP8 claims evaluated: {(claims["claim_category"] != "other").sum():,}")
    print(f"Raw evidence records: {len(ledger):,}")
    print(f"Unique semantic evidence records: {len(deduplicate_semantic_evidence(ledger)):,}")
    print(f"Conflicting evidence records: {len(conflicts):,}")
    print("\nOutputs:")
    for filename in (
        "cross_wp_evidence_ledger.csv",
        "cross_wp_consistency_summary.csv",
        "cross_wp_conflicts.csv",
        "cross_wp_input_manifest.csv",
        "cross_wp_run_summary.csv",
    ):
        print(f"  - {OUTPUT_DIR / filename}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"\nWP9a4 failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        raise
