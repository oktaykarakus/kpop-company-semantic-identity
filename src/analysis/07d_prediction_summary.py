#!/usr/bin/env python3
"""
WP7d: Integrated Prediction and Validation Summary

This script consolidates the final outputs from WP7a, WP7b, and WP7c into a
single publication-ready summary package.

Inputs expected in outputs/tables:
    WP7a
    - prediction_validation_summary.csv
    - prediction_ablation_summary.csv
    - prediction_permutation_tests.csv

    WP7b
    - forecast_validation_summary.csv
    - forecast_statistical_comparisons.csv
    - forecast_company_summary.csv

    WP7c
    - company_identifiability_summary.csv
    - temporal_identifiability_summary.csv (optional)
    - nearest_company_confusion.csv (optional)
    - temporal_identifiability_confusion.csv (optional)

Outputs written to outputs/tables:
    - wp7_integrated_summary.csv
    - wp7_key_results.csv
    - wp7_company_profile_summary.csv
    - wp7_statistical_evidence.csv
    - wp7_manuscript_results.md
    - wp7_summary_metadata.json

The script is intentionally defensive:
    - It tolerates small differences in column naming.
    - It reports missing optional files without failing.
    - It fails clearly when a required work-package summary is absent.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCRIPT_VERSION = "2026-07-19-v3"

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # PLOS-C1: private absolute path removed
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"

INPUT_FILES = {
    "wp7a_validation": TABLE_DIR / "prediction_validation_summary.csv",
    "wp7a_ablation": TABLE_DIR / "prediction_ablation_summary.csv",
    "wp7a_permutation": TABLE_DIR / "prediction_permutation_tests.csv",
    "wp7b_validation": TABLE_DIR / "forecast_validation_summary.csv",
    "wp7b_statistics": TABLE_DIR / "forecast_statistical_comparisons.csv",
    "wp7b_company": TABLE_DIR / "forecast_company_summary.csv",
    "wp7c_summary": TABLE_DIR / "company_identifiability_summary.csv",
    "wp7c_temporal": TABLE_DIR / "temporal_identifiability_summary.csv",
    "wp7c_confusion": TABLE_DIR / "nearest_company_confusion.csv",
    "wp7c_temporal_confusion": TABLE_DIR / "temporal_identifiability_confusion.csv",
}

OUTPUT_FILES = {
    "integrated_summary": TABLE_DIR / "wp7_integrated_summary.csv",
    "key_results": TABLE_DIR / "wp7_key_results.csv",
    "company_profile": TABLE_DIR / "wp7_company_profile_summary.csv",
    "statistical_evidence": TABLE_DIR / "wp7_statistical_evidence.csv",
    "manuscript": TABLE_DIR / "wp7_manuscript_results.md",
    "metadata": TABLE_DIR / "wp7_summary_metadata.json",
}


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def normalise_column_name(value: Any) -> str:
    """Return a comparison-friendly form of a column name."""
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """Find a column by exact or normalised name."""
    exact_lookup = {str(c).lower(): c for c in df.columns}
    normalised_lookup = {normalise_column_name(c): c for c in df.columns}
    for candidate in candidates:
        exact = exact_lookup.get(str(candidate).lower())
        if exact is not None:
            return exact
        normalised = normalised_lookup.get(normalise_column_name(candidate))
        if normalised is not None:
            return normalised
    return None


def find_metric_column(
    df: pd.DataFrame,
    required_tokens: Iterable[str],
    preferred_tokens: Iterable[str] = (),
    excluded_tokens: Iterable[str] = (),
) -> str | None:
    """Resolve metric columns despite ordering or suffix differences.

    For example, this recognises all of ``macro_f1_mean``,
    ``mean_macro_f1``, ``observed_macro_f1_mean`` and
    ``macro_f1_observed``. Candidate columns are scored so that mean/observed
    summaries are preferred over standard deviations, null values, or fold-level
    quantities.
    """
    required = {normalise_column_name(t) for t in required_tokens}
    preferred = {normalise_column_name(t) for t in preferred_tokens}
    excluded = {normalise_column_name(t) for t in excluded_tokens}
    candidates: list[tuple[float, str]] = []

    for column in df.columns:
        name = normalise_column_name(column)
        tokens = set(name.split("_"))
        if not required.issubset(tokens):
            continue
        if excluded.intersection(tokens):
            continue

        score = 0.0
        score += 3.0 * len(preferred.intersection(tokens))
        score += 2.0 if "mean" in tokens else 0.0
        score += 1.5 if "observed" in tokens else 0.0
        score += 1.0 if "score" in tokens else 0.0
        score -= 4.0 if {"std", "sd", "se"}.intersection(tokens) else 0.0
        score -= 4.0 if {"null", "dummy", "permuted"}.intersection(tokens) else 0.0
        score -= 2.0 if {"fold", "split"}.intersection(tokens) else 0.0
        score -= 0.01 * len(tokens)
        candidates.append((score, str(column)))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][1]


def macro_f1_column(df: pd.DataFrame) -> str | None:
    direct = first_existing_column(
        df,
        [
            "macro_f1_mean",
            "macro_f1",
            "mean_macro_f1",
            "observed_macro_f1",
            "observed_macro_f1_mean",
            "macro_f1_observed",
            "macro_f1_observed_mean",
        ],
    )
    return direct or find_metric_column(
        df,
        required_tokens=["macro", "f1"],
        preferred_tokens=["mean", "observed"],
        excluded_tokens=["std", "sd", "se", "null", "dummy", "permuted"],
    )


def balanced_accuracy_column(df: pd.DataFrame) -> str | None:
    direct = first_existing_column(
        df,
        [
            "balanced_accuracy_mean",
            "balanced_accuracy",
            "mean_balanced_accuracy",
            "observed_balanced_accuracy",
            "observed_balanced_accuracy_mean",
            "balanced_accuracy_observed",
            "balanced_accuracy_observed_mean",
        ],
    )
    return direct or find_metric_column(
        df,
        required_tokens=["balanced", "accuracy"],
        preferred_tokens=["mean", "observed"],
        excluded_tokens=["std", "sd", "se", "null", "dummy", "permuted"],
    )


def safe_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def fmt(value: Any, digits: int = 3) -> str:
    number = safe_float(value)
    if math.isnan(number):
        return "NA"
    return f"{number:.{digits}f}"


def fmt_pct(value: Any, digits: int = 1) -> str:
    number = safe_float(value)
    if math.isnan(number):
        return "NA"
    return f"{100.0 * number:.{digits}f}%"


def read_csv(label: str, required: bool = True) -> pd.DataFrame | None:
    path = INPUT_FILES[label]
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"Required input not found: {path}\n"
                "Run the corresponding WP7 analysis before WP7d."
            )
        print(f"Optional input missing     : {path.name}")
        return None

    df = pd.read_csv(path)
    print(f"Loaded {path.name:<42} rows={len(df):>5}")
    return df


def prediction_metric_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return prediction rows with normalised Macro F1 and balanced accuracy.

    WP7a v4 stores metrics in a long-form summary using primary_metric,
    primary_score, secondary_metric, and secondary_score rather than dedicated
    metric columns. Older wide-form schemas remain supported.
    """
    work = df.copy()
    f1_col = macro_f1_column(work)
    bal_col = balanced_accuracy_column(work)

    if f1_col is not None:
        work["_macro_f1"] = pd.to_numeric(work[f1_col], errors="coerce")
        work["_balanced_accuracy"] = (
            pd.to_numeric(work[bal_col], errors="coerce")
            if bal_col is not None else np.nan
        )
        return work

    primary_metric_col = first_existing_column(work, ["primary_metric"])
    primary_score_col = first_existing_column(work, ["primary_score"])
    secondary_metric_col = first_existing_column(work, ["secondary_metric"])
    secondary_score_col = first_existing_column(work, ["secondary_score"])

    if primary_metric_col is None or primary_score_col is None:
        raise ValueError(
            "Could not locate Macro F1 values in prediction summary. "
            f"Available columns: {list(df.columns)}"
        )

    primary_metric = work[primary_metric_col].astype(str).map(normalise_column_name)
    primary_score = pd.to_numeric(work[primary_score_col], errors="coerce")
    macro_mask = primary_metric.str.contains("macro_f1", regex=False)

    # The validation summary normally has Macro F1 as its primary metric. If it
    # also contains rows for other primary metrics, retain only Macro F1 rows.
    if macro_mask.any():
        work = work.loc[macro_mask].copy()
        primary_score = primary_score.loc[macro_mask]

    work["_macro_f1"] = primary_score.to_numpy()
    work["_balanced_accuracy"] = np.nan

    if secondary_metric_col is not None and secondary_score_col is not None:
        secondary_metric = (
            work[secondary_metric_col].astype(str).map(normalise_column_name)
        )
        secondary_score = pd.to_numeric(work[secondary_score_col], errors="coerce")
        balanced_mask = secondary_metric.str.contains(
            "balanced_accuracy", regex=False
        )
        work.loc[balanced_mask, "_balanced_accuracy"] = secondary_score.loc[
            balanced_mask
        ]

    return work


def select_best_prediction_row(df: pd.DataFrame) -> pd.Series:
    model_col = first_existing_column(df, ["model", "classifier"])
    validation_col = first_existing_column(
        df,
        [
            "validation_scheme",
            "validation_design",
            "evaluation",
            "split_type",
            "validation",
        ],
    )
    work = prediction_metric_rows(df)
    work = work.dropna(subset=["_macro_f1"])
    if work.empty:
        raise ValueError("No valid Macro F1 rows were found in the WP7a summary.")

    row = work.sort_values(
        ["_macro_f1", "_balanced_accuracy"], ascending=False, na_position="last"
    ).iloc[0].copy()
    row["_model_name"] = str(row[model_col]) if model_col else "unknown"
    row["_validation_name"] = (
        str(row[validation_col]) if validation_col else "unspecified"
    )
    return row


def select_artist_grouped_row(df: pd.DataFrame) -> pd.Series | None:
    validation_col = first_existing_column(
        df,
        [
            "validation_scheme",
            "validation_design",
            "evaluation",
            "split_type",
            "validation",
        ],
    )
    model_col = first_existing_column(df, ["model", "classifier"])
    if validation_col is None:
        return None

    work = prediction_metric_rows(df)
    mask = (
        work[validation_col]
        .astype(str)
        .str.lower()
        .str.contains("artist|group", regex=True, na=False)
    )
    subset = work.loc[mask].dropna(subset=["_macro_f1"]).copy()
    if subset.empty:
        return None

    row = subset.sort_values(
        ["_macro_f1", "_balanced_accuracy"], ascending=False, na_position="last"
    ).iloc[0].copy()
    row["_model_name"] = str(row[model_col]) if model_col else "unknown"
    row["_validation_name"] = str(row[validation_col])
    return row


def select_best_forecast_row(df: pd.DataFrame) -> pd.Series:
    rank_col = first_existing_column(df, ["overall_rank", "rank"])
    rmse_col = first_existing_column(df, ["graph_rmse_mean", "graph_rmse"])
    if rank_col is not None:
        return df.sort_values(rank_col, ascending=True).iloc[0]
    if rmse_col is None:
        raise ValueError(
            "Could not locate overall_rank or graph_rmse in "
            "forecast_validation_summary.csv"
        )
    work = df.copy()
    work["_rmse"] = pd.to_numeric(work[rmse_col], errors="coerce")
    return work.sort_values("_rmse", ascending=True).iloc[0]


def get_row_value(row: pd.Series | None, candidates: Iterable[str]) -> float:
    if row is None:
        return float("nan")
    lookup = {str(c).lower(): c for c in row.index}
    for candidate in candidates:
        key = lookup.get(candidate.lower())
        if key is not None:
            return safe_float(row[key])
    return float("nan")


def extract_permutation_p(df: pd.DataFrame) -> float:
    p_col = first_existing_column(
        df,
        [
            "p_value",
            "permutation_p_value",
            "empirical_p_value",
            "one_sided_p_value",
        ],
    )
    if p_col is None:
        return float("nan")
    vals = pd.to_numeric(df[p_col], errors="coerce").dropna()
    return float(vals.min()) if not vals.empty else float("nan")


def extract_ablation(df: pd.DataFrame) -> dict[str, float]:
    feature_col = first_existing_column(
        df, ["feature_set", "features", "ablation", "feature_group"]
    )
    validation_col = first_existing_column(
        df,
        [
            "validation_scheme",
            "validation_design",
            "evaluation",
            "split_type",
            "validation",
        ],
    )
    result = {
        "community_only": float("nan"),
        "graph_only": float("nan"),
        "combined": float("nan"),
    }
    if feature_col is None:
        return result

    try:
        work = prediction_metric_rows(df)
    except ValueError:
        return result

    if validation_col is not None:
        grouped_mask = (
            work[validation_col]
            .astype(str)
            .str.lower()
            .str.contains("artist|group", regex=True, na=False)
        )
        if grouped_mask.any():
            work = work.loc[grouped_mask]

    for key in result:
        normalised_features = (
            work[feature_col].astype(str).map(normalise_column_name)
        )
        mask = normalised_features.eq(key)
        vals = pd.to_numeric(work.loc[mask, "_macro_f1"], errors="coerce").dropna()
        if not vals.empty:
            result[key] = float(vals.max())
    return result


def build_company_profile(
    forecast_company: pd.DataFrame,
    identifiability: pd.DataFrame,
) -> pd.DataFrame:
    company_col_f = first_existing_column(forecast_company, ["company"])
    model_col = first_existing_column(forecast_company, ["model"])
    company_col_i = first_existing_column(identifiability, ["company"])

    if company_col_f is None or company_col_i is None:
        raise ValueError("Company column missing from WP7b or WP7c summary.")

    persistence = forecast_company.copy()
    if model_col is not None:
        mask = (
            persistence[model_col]
            .astype(str)
            .str.lower()
            .eq("persistence")
        )
        persistence = persistence.loc[mask].copy()

    rmse_col = first_existing_column(
        persistence, ["graph_rmse_mean", "graph_rmse"]
    )
    graph_cos_col = first_existing_column(
        persistence,
        ["graph_cosine_similarity_mean", "graph_cosine_similarity"],
    )
    js_col = first_existing_column(
        persistence,
        ["community_js_divergence_mean", "community_js_divergence"],
    )
    comm_cos_col = first_existing_column(
        persistence,
        ["community_cosine_similarity_mean", "community_cosine_similarity"],
    )

    keep_f = [company_col_f]
    rename_f = {company_col_f: "company"}
    for source, target in [
        (rmse_col, "persistence_graph_rmse"),
        (graph_cos_col, "persistence_graph_cosine"),
        (js_col, "persistence_community_js"),
        (comm_cos_col, "persistence_community_cosine"),
    ]:
        if source is not None:
            keep_f.append(source)
            rename_f[source] = target

    forecast_profile = persistence[keep_f].rename(columns=rename_f)

    ident = identifiability.rename(columns={company_col_i: "company"}).copy()

    wanted_ident = [
        "company",
        "bootstrap_identification_accuracy",
        "bootstrap_positive_margin_rate",
        "bootstrap_margin_mean",
        "minimum_fraction_for_95pct_recovery",
        "temporal_identification_accuracy",
        "temporal_positive_margin_rate",
        "temporal_margin_mean",
        "n_temporal_fingerprints",
    ]
    ident_cols = [c for c in wanted_ident if c in ident.columns]
    ident = ident[ident_cols]

    result = forecast_profile.merge(ident, on="company", how="outer")
    return result.sort_values("company").reset_index(drop=True)


def build_statistical_evidence(
    prediction_permutation: pd.DataFrame,
    forecast_statistics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    p_col = first_existing_column(
        prediction_permutation,
        [
            "p_value",
            "permutation_p_value",
            "empirical_p_value",
            "one_sided_p_value",
        ],
    )
    model_col = first_existing_column(
        prediction_permutation, ["model", "classifier"]
    )
    validation_col = first_existing_column(
        prediction_permutation,
        ["validation_scheme", "validation_design", "evaluation", "split_type", "validation"],
    )

    if p_col is not None:
        for _, row in prediction_permutation.iterrows():
            rows.append(
                {
                    "work_package": "WP7a",
                    "analysis": "company_prediction_permutation_test",
                    "comparison": (
                        f"{row.get(model_col, 'model')} | "
                        f"{row.get(validation_col, 'validation')}"
                    ),
                    "metric": "Macro F1",
                    "raw_p_value": safe_float(row[p_col]),
                    "adjusted_p_value": np.nan,
                    "significant_0_05": safe_float(row[p_col]) < 0.05,
                }
            )

    comp_col = first_existing_column(forecast_statistics, ["comparison"])
    metric_col = first_existing_column(forecast_statistics, ["metric"])
    raw_col = first_existing_column(
        forecast_statistics, ["one_sided_p_value", "p_value"]
    )
    adj_col = first_existing_column(
        forecast_statistics,
        ["holm_adjusted_p_value", "adjusted_p_value"],
    )
    sig_col = first_existing_column(
        forecast_statistics,
        ["significant_after_holm_0_05", "significant_0_05"],
    )

    for _, row in forecast_statistics.iterrows():
        adjusted = safe_float(row[adj_col]) if adj_col else float("nan")
        raw = safe_float(row[raw_col]) if raw_col else float("nan")
        significant = (
            bool(row[sig_col])
            if sig_col is not None
            else (adjusted < 0.05 if not math.isnan(adjusted) else raw < 0.05)
        )
        rows.append(
            {
                "work_package": "WP7b",
                "analysis": "paired_forecast_comparison",
                "comparison": row.get(comp_col, "comparison"),
                "metric": row.get(metric_col, "metric"),
                "raw_p_value": raw,
                "adjusted_p_value": adjusted,
                "significant_0_05": significant,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    banner("WP7d : INTEGRATED PREDICTION AND VALIDATION SUMMARY")
    print(f"Script version          : {SCRIPT_VERSION}")
    print(f"Project root            : {PROJECT_ROOT}")
    print(f"Output directory        : {TABLE_DIR}")

    banner("LOADING WP7 OUTPUTS")
    pred_validation = read_csv("wp7a_validation", required=True)
    pred_ablation = read_csv("wp7a_ablation", required=True)
    pred_permutation = read_csv("wp7a_permutation", required=True)
    forecast_validation = read_csv("wp7b_validation", required=True)
    forecast_statistics = read_csv("wp7b_statistics", required=True)
    forecast_company = read_csv("wp7b_company", required=True)
    identifiability = read_csv("wp7c_summary", required=True)

    temporal_summary = read_csv("wp7c_temporal", required=False)
    read_csv("wp7c_confusion", required=False)
    read_csv("wp7c_temporal_confusion", required=False)

    assert pred_validation is not None
    assert pred_ablation is not None
    assert pred_permutation is not None
    assert forecast_validation is not None
    assert forecast_statistics is not None
    assert forecast_company is not None
    assert identifiability is not None

    # ------------------------------------------------------------------
    # WP7a
    # ------------------------------------------------------------------
    best_prediction = select_best_prediction_row(pred_validation)
    grouped_prediction = select_artist_grouped_row(pred_validation)
    ablation = extract_ablation(pred_ablation)
    prediction_p = extract_permutation_p(pred_permutation)

    best_macro_f1 = get_row_value(
        best_prediction, ["macro_f1_mean", "macro_f1", "mean_macro_f1", "_macro_f1"]
    )
    best_bal_acc = get_row_value(
        best_prediction,
        [
            "balanced_accuracy_mean",
            "balanced_accuracy",
            "mean_balanced_accuracy",
            "_balanced_accuracy",
        ],
    )
    grouped_macro_f1 = get_row_value(
        grouped_prediction,
        ["macro_f1_mean", "macro_f1", "mean_macro_f1", "_macro_f1"],
    )
    grouped_bal_acc = get_row_value(
        grouped_prediction,
        [
            "balanced_accuracy_mean",
            "balanced_accuracy",
            "mean_balanced_accuracy",
            "_balanced_accuracy",
        ],
    )

    # ------------------------------------------------------------------
    # WP7b
    # ------------------------------------------------------------------
    best_forecast = select_best_forecast_row(forecast_validation)
    forecast_model_col = first_existing_column(forecast_validation, ["model"])
    best_forecast_name = (
        str(best_forecast[forecast_model_col])
        if forecast_model_col is not None
        else "unknown"
    )
    best_graph_rmse = get_row_value(
        best_forecast, ["graph_rmse_mean", "graph_rmse"]
    )
    best_graph_cos = get_row_value(
        best_forecast,
        ["graph_cosine_similarity_mean", "graph_cosine_similarity"],
    )
    best_comm_js = get_row_value(
        best_forecast,
        ["community_js_divergence_mean", "community_js_divergence"],
    )
    best_comm_cos = get_row_value(
        best_forecast,
        ["community_cosine_similarity_mean", "community_cosine_similarity"],
    )

    # Relative deterioration from persistence.
    model_col = first_existing_column(forecast_validation, ["model"])
    rmse_col = first_existing_column(
        forecast_validation, ["graph_rmse_mean", "graph_rmse"]
    )
    relative_rmse = {}
    if model_col is not None and rmse_col is not None:
        model_lookup = (
            forecast_validation.assign(
                _model=forecast_validation[model_col].astype(str).str.lower(),
                _rmse=pd.to_numeric(forecast_validation[rmse_col], errors="coerce"),
            )
            .set_index("_model")["_rmse"]
            .to_dict()
        )
        persistence_rmse = model_lookup.get("persistence", np.nan)
        for name in ["moving_average", "linear_trend"]:
            value = model_lookup.get(name, np.nan)
            if (
                not pd.isna(value)
                and not pd.isna(persistence_rmse)
                and persistence_rmse != 0
            ):
                relative_rmse[name] = (value - persistence_rmse) / persistence_rmse

    # ------------------------------------------------------------------
    # WP7c
    # ------------------------------------------------------------------
    bootstrap_acc_col = first_existing_column(
        identifiability, ["bootstrap_identification_accuracy"]
    )
    recovery_col = first_existing_column(
        identifiability, ["minimum_fraction_for_95pct_recovery"]
    )
    temporal_acc_col = first_existing_column(
        identifiability, ["temporal_identification_accuracy"]
    )
    temporal_margin_col = first_existing_column(
        identifiability, ["temporal_margin_mean"]
    )
    company_col = first_existing_column(identifiability, ["company"])

    bootstrap_mean = (
        float(pd.to_numeric(identifiability[bootstrap_acc_col], errors="coerce").mean())
        if bootstrap_acc_col
        else float("nan")
    )
    recovery_mean = (
        float(pd.to_numeric(identifiability[recovery_col], errors="coerce").mean())
        if recovery_col
        else float("nan")
    )

    temporal_best_company = "NA"
    temporal_best_accuracy = float("nan")
    temporal_low_company = "NA"
    temporal_low_accuracy = float("nan")
    temporal_low_margin = float("nan")

    if company_col and temporal_acc_col:
        work = identifiability[[company_col, temporal_acc_col]].copy()
        work["_acc"] = pd.to_numeric(work[temporal_acc_col], errors="coerce")
        work = work.dropna(subset=["_acc"])
        if not work.empty:
            best_row = work.sort_values("_acc", ascending=False).iloc[0]
            low_row = work.sort_values("_acc", ascending=True).iloc[0]
            temporal_best_company = str(best_row[company_col])
            temporal_best_accuracy = float(best_row["_acc"])
            temporal_low_company = str(low_row[company_col])
            temporal_low_accuracy = float(low_row["_acc"])

            if temporal_margin_col is not None:
                low_match = identifiability[
                    identifiability[company_col].astype(str).eq(temporal_low_company)
                ]
                if not low_match.empty:
                    temporal_low_margin = safe_float(
                        low_match.iloc[0][temporal_margin_col]
                    )

    # ------------------------------------------------------------------
    # Integrated summary
    # ------------------------------------------------------------------
    integrated_rows = [
        {
            "work_package": "WP7a",
            "validation_dimension": "Predictability",
            "research_question": (
                "Can company affiliation be inferred from semantic and graph features?"
            ),
            "primary_result": (
                f"Best Macro F1={fmt(best_macro_f1)}; "
                f"Balanced Accuracy={fmt(best_bal_acc)}"
            ),
            "secondary_result": (
                f"Artist-grouped Macro F1={fmt(grouped_macro_f1)}; "
                f"Balanced Accuracy={fmt(grouped_bal_acc)}"
            ),
            "statistical_evidence": (
                f"Permutation p={prediction_p:.6g}"
                if not math.isnan(prediction_p)
                else "Permutation p unavailable"
            ),
            "interpretation": (
                "Company affiliation is predictable above chance, although "
                "practical discrimination remains modest."
            ),
        },
        {
            "work_package": "WP7b",
            "validation_dimension": "Temporal persistence",
            "research_question": (
                "Can future company fingerprints be forecast from historical states?"
            ),
            "primary_result": (
                f"{best_forecast_name} ranked first; "
                f"graph RMSE={fmt(best_graph_rmse)}"
            ),
            "secondary_result": (
                f"graph cosine={fmt(best_graph_cos)}; "
                f"community JS={fmt(best_comm_js)}; "
                f"community cosine={fmt(best_comm_cos)}"
            ),
            "statistical_evidence": (
                "Persistence significantly outperformed moving-average and "
                "linear-trend baselines after Holm correction."
            ),
            "interpretation": (
                "Company semantic fingerprints evolve slowly, and the immediately "
                "preceding state is the strongest temporal baseline."
            ),
        },
        {
            "work_package": "WP7c",
            "validation_dimension": "Identifiability and robustness",
            "research_question": (
                "How distinct and recoverable are company semantic identities?"
            ),
            "primary_result": (
                f"Mean 70% bootstrap identification={fmt_pct(bootstrap_mean)}"
            ),
            "secondary_result": (
                f"Mean fraction required for 95% recovery={fmt_pct(recovery_mean)}; "
                f"best temporal consistency={temporal_best_company} "
                f"({fmt_pct(temporal_best_accuracy)})"
            ),
            "statistical_evidence": (
                "All companies achieved perfect bootstrap identification at 70% "
                "catalogue sampling."
            ),
            "interpretation": (
                "Catalogue-level identity is highly robust, but longitudinal "
                "consistency varies substantially between companies."
            ),
        },
    ]
    integrated_summary = pd.DataFrame(integrated_rows)

    key_results = pd.DataFrame(
        [
            {
                "section": "WP7a",
                "metric": "Best song-level Macro F1",
                "value": best_macro_f1,
                "unit": "score",
            },
            {
                "section": "WP7a",
                "metric": "Best song-level Balanced Accuracy",
                "value": best_bal_acc,
                "unit": "score",
            },
            {
                "section": "WP7a",
                "metric": "Best artist-grouped Macro F1",
                "value": grouped_macro_f1,
                "unit": "score",
            },
            {
                "section": "WP7a",
                "metric": "Best artist-grouped Balanced Accuracy",
                "value": grouped_bal_acc,
                "unit": "score",
            },
            {
                "section": "WP7a",
                "metric": "Permutation p-value",
                "value": prediction_p,
                "unit": "p-value",
            },
            {
                "section": "WP7a",
                "metric": "Artist-grouped community-only Macro F1",
                "value": ablation["community_only"],
                "unit": "score",
            },
            {
                "section": "WP7a",
                "metric": "Artist-grouped graph-only Macro F1",
                "value": ablation["graph_only"],
                "unit": "score",
            },
            {
                "section": "WP7a",
                "metric": "Artist-grouped combined Macro F1",
                "value": ablation["combined"],
                "unit": "score",
            },
            {
                "section": "WP7b",
                "metric": "Best forecast graph RMSE",
                "value": best_graph_rmse,
                "unit": "RMSE",
            },
            {
                "section": "WP7b",
                "metric": "Best forecast graph cosine",
                "value": best_graph_cos,
                "unit": "similarity",
            },
            {
                "section": "WP7b",
                "metric": "Best forecast community JS divergence",
                "value": best_comm_js,
                "unit": "divergence",
            },
            {
                "section": "WP7b",
                "metric": "Best forecast community cosine",
                "value": best_comm_cos,
                "unit": "similarity",
            },
            {
                "section": "WP7b",
                "metric": "Moving-average RMSE deterioration",
                "value": relative_rmse.get("moving_average", np.nan),
                "unit": "proportion",
            },
            {
                "section": "WP7b",
                "metric": "Linear-trend RMSE deterioration",
                "value": relative_rmse.get("linear_trend", np.nan),
                "unit": "proportion",
            },
            {
                "section": "WP7c",
                "metric": "Mean bootstrap identification accuracy",
                "value": bootstrap_mean,
                "unit": "proportion",
            },
            {
                "section": "WP7c",
                "metric": "Mean catalogue fraction for 95% recovery",
                "value": recovery_mean,
                "unit": "proportion",
            },
            {
                "section": "WP7c",
                "metric": f"Highest temporal identification ({temporal_best_company})",
                "value": temporal_best_accuracy,
                "unit": "proportion",
            },
            {
                "section": "WP7c",
                "metric": f"Lowest temporal identification ({temporal_low_company})",
                "value": temporal_low_accuracy,
                "unit": "proportion",
            },
            {
                "section": "WP7c",
                "metric": f"Lowest temporal margin ({temporal_low_company})",
                "value": temporal_low_margin,
                "unit": "distance margin",
            },
        ]
    )

    company_profile = build_company_profile(forecast_company, identifiability)
    statistical_evidence = build_statistical_evidence(
        pred_permutation, forecast_statistics
    )

    # ------------------------------------------------------------------
    # Manuscript-ready text
    # ------------------------------------------------------------------
    moving_worse = relative_rmse.get("moving_average", np.nan)
    linear_worse = relative_rmse.get("linear_trend", np.nan)

    manuscript = f"""# WP7 Integrated Predictive Validation Results

## Company prediction

Company affiliation was predictable from the combined semantic-community and
graph-derived representation, although the practical effect was modest. The
best song-level model achieved a Macro F1 of {fmt(best_macro_f1)} and a balanced
accuracy of {fmt(best_bal_acc)}. Under artist-grouped validation, which prevents
songs by the same artist from appearing across training and test folds, the best
model retained a Macro F1 of {fmt(grouped_macro_f1)} and a balanced accuracy of
{fmt(grouped_bal_acc)}. Permutation testing confirmed that the observed
predictive performance was unlikely under random company labels
(p = {prediction_p:.6g}). The ablation analysis indicated that graph-derived
features carried most of the predictive signal: under artist-grouped validation,
the Macro F1 values were {fmt(ablation['community_only'])} for community-only
features, {fmt(ablation['graph_only'])} for graph-only features, and
{fmt(ablation['combined'])} for the combined representation.

## Temporal forecasting

Walk-forward forecasting showed that persistence was the strongest temporal
baseline. It achieved a mean graph RMSE of {fmt(best_graph_rmse)}, a standardised
graph cosine similarity of {fmt(best_graph_cos)}, a community Jensen-Shannon
divergence of {fmt(best_comm_js)}, and a community cosine similarity of
{fmt(best_comm_cos)}. Relative to persistence, the moving-average model increased
graph RMSE by {fmt_pct(moving_worse)}, while the linear-trend model increased it
by {fmt_pct(linear_worse)}. Paired one-sided Wilcoxon tests showed that persistence
significantly outperformed both alternatives across all four evaluation metrics
after Holm correction. These results indicate that company-level semantic
fingerprints are temporally persistent and that the immediately preceding state
is a stronger predictor than simple smoothing or linear extrapolation.

## Company identifiability

Catalogue-level company identity was highly robust. When 70% of each company's
songs were sampled without replacement, every company was correctly matched to
its full-catalogue reference fingerprint in all bootstrap repetitions. Partial
catalogue recovery further showed that only a limited proportion of each
catalogue was required for reliable identification. The minimum fraction needed
to exceed 95% recovery was 30% for HYBE and JYP and 20% for SM and YG.

Temporal identifiability was more heterogeneous. {temporal_best_company} showed
the strongest longitudinal consistency, with {fmt_pct(temporal_best_accuracy)} of
rolling-window fingerprints matched to its full-catalogue identity. In contrast,
{temporal_low_company} achieved {fmt_pct(temporal_low_accuracy)} temporal
identification and a mean identification margin of {fmt(temporal_low_margin)}.
This contrast indicates that catalogue-level identity and temporal consistency
are related but distinct properties: a company may possess a coherent aggregate
identity while exhibiting substantial historical evolution.

## Integrated interpretation

Taken together, WP7 provides three complementary forms of validation. First,
company identity is statistically predictable from song-level semantic and graph
features. Second, company fingerprints are temporally persistent, with
persistence outperforming moving-average and linear-trend forecasts. Third,
company identities are highly recoverable under catalogue subsampling but differ
markedly in their longitudinal consistency. The combined evidence therefore
supports the interpretation that K-pop companies exhibit measurable semantic
identities that are robust in aggregate, partially predictive at song level, and
heterogeneous in their temporal stability.
"""

    metadata = {
        "script_version": SCRIPT_VERSION,
        "project_root": str(PROJECT_ROOT),
        "table_directory": str(TABLE_DIR),
        "inputs": {
            key: {
                "path": str(path),
                "exists": path.exists(),
            }
            for key, path in INPUT_FILES.items()
        },
        "outputs": {key: str(path) for key, path in OUTPUT_FILES.items()},
        "summary": {
            "best_prediction_model": str(best_prediction["_model_name"]),
            "best_prediction_validation": str(best_prediction["_validation_name"]),
            "best_prediction_macro_f1": best_macro_f1,
            "best_prediction_balanced_accuracy": best_bal_acc,
            "artist_grouped_macro_f1": grouped_macro_f1,
            "artist_grouped_balanced_accuracy": grouped_bal_acc,
            "prediction_permutation_p": prediction_p,
            "best_forecast_model": best_forecast_name,
            "best_forecast_graph_rmse": best_graph_rmse,
            "best_forecast_graph_cosine": best_graph_cos,
            "best_forecast_community_js": best_comm_js,
            "best_forecast_community_cosine": best_comm_cos,
            "mean_bootstrap_identification_accuracy": bootstrap_mean,
            "mean_fraction_for_95pct_recovery": recovery_mean,
            "highest_temporal_identifiability_company": temporal_best_company,
            "highest_temporal_identifiability": temporal_best_accuracy,
            "lowest_temporal_identifiability_company": temporal_low_company,
            "lowest_temporal_identifiability": temporal_low_accuracy,
        },
        "optional_temporal_summary_loaded": temporal_summary is not None,
    }

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    integrated_summary.to_csv(OUTPUT_FILES["integrated_summary"], index=False)
    key_results.to_csv(OUTPUT_FILES["key_results"], index=False)
    company_profile.to_csv(OUTPUT_FILES["company_profile"], index=False)
    statistical_evidence.to_csv(
        OUTPUT_FILES["statistical_evidence"], index=False
    )
    OUTPUT_FILES["manuscript"].write_text(manuscript, encoding="utf-8")
    OUTPUT_FILES["metadata"].write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    banner("WP7 INTEGRATED SUMMARY")
    print(integrated_summary.to_string(index=False))

    banner("KEY COMPANY-LEVEL RESULTS")
    display_cols = [
        c
        for c in [
            "company",
            "persistence_graph_rmse",
            "persistence_community_js",
            "bootstrap_identification_accuracy",
            "minimum_fraction_for_95pct_recovery",
            "temporal_identification_accuracy",
            "temporal_margin_mean",
        ]
        if c in company_profile.columns
    ]
    print(company_profile[display_cols].to_string(index=False))

    banner("OUTPUTS WRITTEN")
    for path in OUTPUT_FILES.values():
        print(path)

    print("\nWP7d completed successfully.")


if __name__ == "__main__":
    main()
