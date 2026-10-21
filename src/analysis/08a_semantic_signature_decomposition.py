#!/usr/bin/env python3
"""
WP8a: Semantic Signature Decomposition

Scientific question
-------------------
Which semantic dimensions jointly define each entertainment company's
organisational identity?

Inputs
------
outputs/tables/company_master_analysis_v1.parquet
outputs/temporal/rolling5/semantic_drift_summary.csv
outputs/tables/wp7_company_profile_summary.csv

Outputs
-------
outputs/explainability/company_signature_components.csv
outputs/explainability/company_signature_summary.csv
outputs/explainability/company_signature_rankings.csv
outputs/explainability/signature_strength.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_VERSION = "2026-07-19-v2"
ROOT = Path(__file__).resolve().parents[2]

MASTER_PATH = ROOT / "outputs" / "tables" / "company_master_analysis_v1.parquet"
DRIFT_SUMMARY_PATH = ROOT / "outputs" / "temporal" / "rolling5" / "semantic_drift_summary.csv"
WP7_COMPANY_PATH = ROOT / "outputs" / "tables" / "wp7_company_profile_summary.csv"

OUTPUT_DIR = ROOT / "outputs" / "explainability"
COMPONENTS_PATH = OUTPUT_DIR / "company_signature_components.csv"
SUMMARY_PATH = OUTPUT_DIR / "company_signature_summary.csv"
RANKINGS_PATH = OUTPUT_DIR / "company_signature_rankings.csv"
STRENGTH_PATH = OUTPUT_DIR / "signature_strength.csv"
METADATA_PATH = OUTPUT_DIR / "company_signature_metadata.json"

COMPANY_ORDER = ["HYBE", "JYP", "SM", "YG"]

BASE_METRICS = {
    "degree": ("Semantic connectivity", "higher", "Average number of semantic relationships surrounding songs."),
    "weighted_degree": ("Weighted semantic connectivity", "higher", "Average strength of semantic relationships surrounding songs."),
    "bridge_affinity_score": ("Bridge affinity", "higher", "Tendency to connect otherwise distinct semantic regions."),
    "local_novelty": ("Local novelty", "higher", "Distinctiveness relative to nearby semantic neighbours."),
    "centroid_novelty": ("Centroid novelty", "higher", "Distance from the company's central semantic tendency."),
    "community_entropy": ("Song-level semantic entropy", "higher", "Average semantic spread across communities at song level."),
    "participation_coefficient": ("Cross-community participation", "higher", "Extent to which songs participate across multiple communities."),
}

DERIVED_METRICS = {
    "catalogue_entropy": ("Catalogue semantic entropy", "higher", "Diversity of the company's complete semantic-community distribution."),
    "effective_communities": ("Effective communities", "higher", "Effective number of meaningfully represented semantic communities."),
}

WP7_METRICS = {
    "persistence_graph_rmse": ("Graph forecast error", "lower", "Difficulty of forecasting the next company graph fingerprint."),
    "persistence_graph_cosine": ("Graph temporal persistence", "higher", "Similarity between consecutive graph fingerprints."),
    "persistence_community_js": ("Community forecast divergence", "lower", "Change in semantic-community composition across adjacent periods."),
    "persistence_community_cosine": ("Community temporal persistence", "higher", "Similarity between consecutive community fingerprints."),
    "bootstrap_identification_accuracy": ("Bootstrap identifiability", "higher", "Recoverability of company identity from a partial catalogue."),
    "minimum_fraction_for_95pct_recovery": ("Catalogue requirement for recovery", "lower", "Catalogue fraction needed to recover company identity reliably."),
    "temporal_identification_accuracy": ("Temporal identifiability", "higher", "Consistency with which historical fingerprints match the full company identity."),
    "temporal_margin_mean": ("Temporal identity margin", "higher", "Separation between a company's historical identity and its nearest competitor."),
}

DRIFT_CANDIDATES = {
    "mean_semantic_drift": ["mean_js", "mean_semantic_drift", "semantic_drift_mean", "mean_js_divergence", "mean_drift", "average_semantic_drift"],
    "median_semantic_drift": ["median_js", "median_semantic_drift", "semantic_drift_median", "median_js_divergence", "median_drift"],
    "maximum_semantic_drift": ["max_js", "max_semantic_drift", "maximum_semantic_drift", "semantic_drift_max", "max_js_divergence", "maximum_drift"],
    "semantic_stability": ["mean_cosine", "semantic_stability", "stability_score", "temporal_stability"],
}

DRIFT_METADATA = {
    "mean_semantic_drift": ("Mean semantic drift", "higher", "Average magnitude of semantic change across rolling periods."),
    "median_semantic_drift": ("Median semantic drift", "higher", "Typical magnitude of semantic change across rolling periods."),
    "maximum_semantic_drift": ("Maximum semantic drift", "higher", "Largest observed semantic shift across rolling periods."),
    "semantic_stability": ("Temporal semantic stability", "higher", "Overall persistence of the company fingerprint through time."),
}


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")


def locate_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {str(column).strip().lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def standard_z(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    std = numeric.std(ddof=0)
    if pd.isna(std) or np.isclose(std, 0.0):
        return pd.Series(0.0, index=values.index)
    return (numeric - numeric.mean()) / std


def robust_z(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    median = numeric.median()
    mad = np.median(np.abs(numeric.dropna() - median))
    if pd.isna(mad) or np.isclose(mad, 0.0):
        return standard_z(numeric)
    return 0.6744897501960817 * (numeric - median) / mad


def percentile_rank(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").rank(method="average", pct=True) * 100.0


def qualitative_category(percentile: float) -> str:
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


def company_rank(values: pd.Series, direction: str) -> pd.Series:
    ascending = direction == "lower"
    return pd.to_numeric(values, errors="coerce").rank(method="min", ascending=ascending).astype("Int64")


def summarise_master(master: pd.DataFrame) -> pd.DataFrame:
    required = ["company", "analysis_eligible", "community", *BASE_METRICS.keys()]
    missing = [column for column in required if column not in master.columns]
    if missing:
        raise ValueError("Master dataset is missing required columns: " + ", ".join(missing))

    eligible = master.loc[master["analysis_eligible"].eq(True)].copy()
    if eligible.empty:
        raise ValueError("No analysis-eligible rows were found in the master dataset.")

    means = eligible.groupby("company", observed=True)[list(BASE_METRICS)].mean().reset_index()

    counts = eligible.groupby(["company", "community"], observed=True).size().rename("n_songs").reset_index()
    counts["community_probability"] = counts["n_songs"] / counts.groupby("company", observed=True)["n_songs"].transform("sum")
    counts["entropy_component"] = np.where(
        counts["community_probability"] > 0,
        -counts["community_probability"] * np.log(counts["community_probability"]),
        0.0,
    )
    entropy = counts.groupby("company", observed=True)["entropy_component"].sum().rename("catalogue_entropy").reset_index()
    entropy["effective_communities"] = np.exp(entropy["catalogue_entropy"])

    return means.merge(entropy, on="company", how="left")


def summarise_drift(drift: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, tuple[str, str, str]]]:
    company_column = locate_column(drift, ["company"])
    if company_column is None:
        raise ValueError(f"semantic_drift_summary.csv has no company column. Available columns: {list(drift.columns)}")

    selected = pd.DataFrame({"company": drift[company_column].astype(str)})
    metadata: dict[str, tuple[str, str, str]] = {}

    for output_name, candidates in DRIFT_CANDIDATES.items():
        source = locate_column(drift, candidates)
        if source is not None:
            selected[output_name] = pd.to_numeric(drift[source], errors="coerce")
            metadata[output_name] = DRIFT_METADATA[output_name]

    if len(selected.columns) == 1:
        raise ValueError(f"No recognised drift metric found. Available columns: {list(drift.columns)}")

    return selected, metadata


def summarise_wp7(wp7: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, tuple[str, str, str]]]:
    if "company" not in wp7.columns:
        raise ValueError("wp7_company_profile_summary.csv has no company column.")

    columns = ["company"]
    metadata: dict[str, tuple[str, str, str]] = {}
    for metric, details in WP7_METRICS.items():
        if metric in wp7.columns:
            columns.append(metric)
            metadata[metric] = details

    if len(columns) == 1:
        raise ValueError(f"No recognised WP7 company metric found. Available columns: {list(wp7.columns)}")

    selected = wp7[columns].copy()
    for column in columns[1:]:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    return selected, metadata


def build_components(company_metrics: pd.DataFrame, metadata: dict[str, tuple[str, str, str]]) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for metric, (label, direction, description) in metadata.items():
        values = company_metrics.set_index("company")[metric]
        z_scores = standard_z(values)
        robust_scores = robust_z(values)
        percentiles = percentile_rank(values)
        rankings = company_rank(values, direction)

        for company in values.index:
            percentile = float(percentiles.loc[company])
            records.append({
                "company": company,
                "metric": metric,
                "semantic_dimension": label,
                "metric_description": description,
                "interpretive_direction": direction,
                "raw_value": float(values.loc[company]) if pd.notna(values.loc[company]) else np.nan,
                "z_score": float(z_scores.loc[company]) if pd.notna(z_scores.loc[company]) else np.nan,
                "robust_z_score": float(robust_scores.loc[company]) if pd.notna(robust_scores.loc[company]) else np.nan,
                "percentile": percentile,
                "ranking": int(rankings.loc[company]) if pd.notna(rankings.loc[company]) else pd.NA,
                "qualitative_category": qualitative_category(percentile),
            })

    components = pd.DataFrame(records)
    components["ranking"] = components["ranking"].astype("Int64")
    return components.sort_values(["company", "ranking", "metric"], kind="stable").reset_index(drop=True)


def build_strength(components: pd.DataFrame) -> pd.DataFrame:
    strength = components.groupby("company", observed=True).agg(
        signature_strength=("robust_z_score", lambda values: float(np.nanmean(np.abs(values)))),
        mean_absolute_z_score=("z_score", lambda values: float(np.nanmean(np.abs(values)))),
        n_signature_components=("metric", "count"),
        n_very_high_components=("qualitative_category", lambda values: int((values == "Very High").sum())),
        n_very_low_components=("qualitative_category", lambda values: int((values == "Very Low").sum())),
    ).reset_index()

    strength["signature_strength_rank"] = strength["signature_strength"].rank(method="min", ascending=False).astype("Int64")
    strength["signature_strength_percentile"] = percentile_rank(strength["signature_strength"])
    strength["signature_strength_category"] = strength["signature_strength_percentile"].map(qualitative_category)
    return strength.sort_values(["signature_strength_rank", "company"]).reset_index(drop=True)


def build_summary(components: pd.DataFrame, strength: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for company in COMPANY_ORDER:
        subset = components.loc[components["company"].eq(company)].copy()
        if subset.empty:
            continue

        strongest = subset.assign(abs_robust_z=subset["robust_z_score"].abs()).sort_values(
            ["abs_robust_z", "metric"], ascending=[False, True]
        )
        high = subset.loc[subset["qualitative_category"].isin(["High", "Very High"])].sort_values(
            ["percentile", "metric"], ascending=[False, True]
        )
        low = subset.loc[subset["qualitative_category"].isin(["Low", "Very Low"])].sort_values(
            ["percentile", "metric"], ascending=[True, True]
        )
        strength_row = strength.loc[strength["company"].eq(company)].iloc[0]

        rows.append({
            "company": company,
            "signature_strength": strength_row["signature_strength"],
            "signature_strength_rank": strength_row["signature_strength_rank"],
            "signature_strength_category": strength_row["signature_strength_category"],
            "most_distinctive_dimension": strongest.iloc[0]["semantic_dimension"],
            "most_distinctive_raw_value": strongest.iloc[0]["raw_value"],
            "most_distinctive_robust_z": strongest.iloc[0]["robust_z_score"],
            "highest_dimensions": "; ".join(high["semantic_dimension"].head(5).tolist()),
            "lowest_dimensions": "; ".join(low["semantic_dimension"].head(5).tolist()),
            "n_high_or_very_high": int(len(high)),
            "n_low_or_very_low": int(len(low)),
        })

    summary = pd.DataFrame(rows)
    summary["signature_strength_rank"] = summary["signature_strength_rank"].astype("Int64")
    return summary.sort_values("signature_strength_rank").reset_index(drop=True)


def main() -> None:
    banner("WP8a : SEMANTIC SIGNATURE DECOMPOSITION")
    print(f"Script version          : {SCRIPT_VERSION}")
    print(f"Project root            : {ROOT}")
    print(f"Master dataset          : {MASTER_PATH}")
    print(f"Temporal drift summary  : {DRIFT_SUMMARY_PATH}")
    print(f"WP7 company profile     : {WP7_COMPANY_PATH}")
    print(f"Output directory        : {OUTPUT_DIR}")

    require_file(MASTER_PATH)
    require_file(DRIFT_SUMMARY_PATH)
    require_file(WP7_COMPANY_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    banner("LOADING EXISTING PIPELINE OUTPUTS")
    master = pd.read_parquet(MASTER_PATH)
    drift = pd.read_csv(DRIFT_SUMMARY_PATH)
    wp7 = pd.read_csv(WP7_COMPANY_PATH)

    print(f"Master rows             : {len(master):,}")
    print(f"Analysis-eligible rows  : {int(master['analysis_eligible'].eq(True).sum()):,}")
    print(f"Drift summary rows      : {len(drift):,}")
    print(f"WP7 company rows        : {len(wp7):,}")

    master_summary = summarise_master(master)
    drift_summary, drift_metadata = summarise_drift(drift)
    wp7_summary, wp7_metadata = summarise_wp7(wp7)

    metric_metadata = {**BASE_METRICS, **DERIVED_METRICS, **drift_metadata, **wp7_metadata}

    company_metrics = master_summary.merge(drift_summary, on="company", how="left").merge(wp7_summary, on="company", how="left")
    company_metrics["company"] = pd.Categorical(company_metrics["company"], categories=COMPANY_ORDER, ordered=True)
    company_metrics = company_metrics.sort_values("company").reset_index(drop=True)
    company_metrics["company"] = company_metrics["company"].astype(str)

    if set(company_metrics["company"]) != set(COMPANY_ORDER):
        raise ValueError(f"Unexpected company set: {sorted(set(company_metrics['company']))}")

    components = build_components(company_metrics, metric_metadata)
    strength = build_strength(components)
    summary = build_summary(components, strength)
    rankings = components[[
        "metric", "semantic_dimension", "company", "raw_value", "robust_z_score",
        "percentile", "ranking", "qualitative_category", "interpretive_direction"
    ]].sort_values(["metric", "ranking", "company"], kind="stable").reset_index(drop=True)

    components.to_csv(COMPONENTS_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    rankings.to_csv(RANKINGS_PATH, index=False)
    strength.to_csv(STRENGTH_PATH, index=False)

    METADATA_PATH.write_text(json.dumps({
        "script_version": SCRIPT_VERSION,
        "inputs": {
            "master": str(MASTER_PATH),
            "semantic_drift_summary": str(DRIFT_SUMMARY_PATH),
            "wp7_company_profile": str(WP7_COMPANY_PATH),
        },
        "outputs": {
            "company_signature_components": str(COMPONENTS_PATH),
            "company_signature_summary": str(SUMMARY_PATH),
            "company_signature_rankings": str(RANKINGS_PATH),
            "signature_strength": str(STRENGTH_PATH),
        },
        "n_companies": int(company_metrics["company"].nunique()),
        "n_metrics": int(len(metric_metadata)),
        "metrics": list(metric_metadata),
        "signature_strength_definition": "mean(abs(robust z-score))",
        "robust_z_definition": "0.6744897501960817 * (x - median) / MAD",
    }, indent=2), encoding="utf-8")

    banner("SIGNATURE STRENGTH")
    print(strength[[
        "company", "signature_strength", "signature_strength_rank",
        "signature_strength_category", "n_signature_components"
    ]].to_string(index=False))

    banner("COMPANY SIGNATURE SUMMARY")
    print(summary[[
        "company", "signature_strength", "signature_strength_rank",
        "most_distinctive_dimension", "highest_dimensions", "lowest_dimensions"
    ]].to_string(index=False))

    banner("OUTPUTS WRITTEN")
    for path in [COMPONENTS_PATH, SUMMARY_PATH, RANKINGS_PATH, STRENGTH_PATH, METADATA_PATH]:
        print(path)

    print("\nWP8a completed successfully.")


if __name__ == "__main__":
    main()
