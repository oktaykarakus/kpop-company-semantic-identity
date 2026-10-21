#!/usr/bin/env python3
"""
07b_temporal_forecasting_v2.py

WP7b - Temporal forecasting of company semantic fingerprints.

Scientific question
-------------------
Do earlier company semantic fingerprints contain enough temporal continuity to
predict the immediately following rolling-window fingerprint?

The script uses only the fixed master dataset:

    outputs/tables/company_master_analysis_v1.parquet

and does not search for alternative files.

Forecast design
---------------
For each company and each rolling-window width (3, 5, and 7 years), the script
constructs annual-endpoint semantic fingerprints and performs strict walk-forward
one-step-ahead forecasting. At each forecast origin, only earlier fingerprints
are available to the model.

Forecast models
---------------
1. persistence
   Predict the next fingerprint as the most recent observed fingerprint.

2. linear_trend
   Extrapolate each fingerprint dimension using an ordinary least-squares trend
   fitted to all available historical fingerprints.

3. moving_average
   Predict using the mean of the most recent three available fingerprints.

Fingerprint components
----------------------
A. Continuous graph fingerprint
   Mean values of the existing graph/novelty variables in each company-window.

B. Semantic-community distribution
   Proportions of songs assigned to each existing semantic community.

Evaluation metrics
------------------
- RMSE on history-standardised continuous graph features (lower is better)
- Cosine similarity after history-only standardisation of graph features (higher is better)
- Jensen-Shannon divergence for the community distribution (lower is better)
- Cosine similarity for the community distribution (higher is better)

Outputs
-------
Written to outputs/tables:

- forecast_results.csv
- forecast_fold_results.csv
- forecast_predictions.csv
- forecast_window_fingerprints.csv
- forecast_model_comparison.csv
- forecast_feature_manifest.csv
- forecast_validation_summary.csv
- forecast_statistical_comparisons.csv
- forecast_company_summary.csv
- forecast_metadata.json

No figures are generated.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wilcoxon


SCRIPT_VERSION = "2026-07-19-v2"
ANALYSIS_NAME = "WP7b : TEMPORAL FORECASTING"

ROOT = Path(__file__).resolve().parents[2]
MASTER_PATH = ROOT / "outputs" / "tables" / "company_master_analysis_v1.parquet"
OUTPUT_DIR = ROOT / "outputs" / "tables"

ELIGIBILITY_COLUMN = "analysis_eligible"
COMPANY_COLUMN = "company"
YEAR_COLUMN = "year"
COMMUNITY_COLUMN = "community"

WINDOW_SIZES = (3, 5, 7)
MIN_SONGS_PER_WINDOW = 12
MIN_DISTINCT_YEARS_PER_WINDOW = 2
MOVING_AVERAGE_HISTORY = 3
EPS = 1e-12

GRAPH_FEATURE_CANDIDATES = [
    "degree",
    "weighted_degree",
    "strength",
    "betweenness_centrality",
    "closeness_centrality",
    "eigenvector_centrality",
    "pagerank",
    "clustering_coefficient",
    "bridge_affinity_score",
    "bridge_score",
    "local_novelty",
    "centroid_novelty",
    "community_entropy",
    "semantic_entropy",
    "community_size",
    "distance_to_community_centroid",
    "distance_to_global_centroid",
    "nearest_neighbour_similarity",
    "mean_neighbour_similarity",
    "within_community_similarity",
    "cross_community_similarity",
    "participation_coefficient",
    "within_module_degree_zscore",
]

MODEL_NAMES = ("persistence", "linear_trend", "moving_average")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def print_section(title: str) -> None:
    line = "=" * 78
    print(f"\n{line}\n{title}\n{line}")


def ensure_required_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            "Required master dataset was not found at the fixed project path:\n"
            f"  {path}\n"
            "WP7 does not search for alternative files."
        )


def existing_columns(frame: pd.DataFrame, candidates: Iterable[str]) -> list[str]:
    return [column for column in candidates if column in frame.columns]


def safe_cosine_similarity(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(predicted)
    if not np.any(mask):
        return float("nan")
    a = actual[mask]
    p = predicted[mask]
    denominator = float(np.linalg.norm(a) * np.linalg.norm(p))
    if denominator <= EPS:
        return float("nan")
    return float(np.dot(a, p) / denominator)


def normalise_probability_vector(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    vector = np.where(np.isfinite(vector), vector, 0.0)
    vector = np.clip(vector, 0.0, None)
    total = float(vector.sum())
    if total <= EPS:
        return np.full(vector.shape, 1.0 / len(vector), dtype=float)
    return vector / total


def safe_js_divergence(actual: np.ndarray, predicted: np.ndarray) -> float:
    a = normalise_probability_vector(actual)
    p = normalise_probability_vector(predicted)
    # scipy returns Jensen-Shannon distance; squaring gives the divergence.
    distance = float(jensenshannon(a, p, base=2.0))
    return distance * distance


def history_standardisation_parameters(history: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    history = np.asarray(history, dtype=float)
    means = np.nanmean(history, axis=0)
    scales = np.nanstd(history, axis=0, ddof=0)
    means = np.where(np.isfinite(means), means, 0.0)
    scales = np.where(np.isfinite(scales) & (scales > EPS), scales, 1.0)
    return means, scales


def history_standardised_vectors(
    actual: np.ndarray,
    predicted: np.ndarray,
    history: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    means, scales = history_standardisation_parameters(history)
    return (np.asarray(actual, dtype=float) - means) / scales, (
        np.asarray(predicted, dtype=float) - means
    ) / scales


def history_standardised_rmse(
    actual: np.ndarray,
    predicted: np.ndarray,
    history: np.ndarray,
) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    history = np.asarray(history, dtype=float)

    a, p = history_standardised_vectors(actual, predicted, history)
    mask = np.isfinite(a) & np.isfinite(p)
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.mean((a[mask] - p[mask]) ** 2)))


def nanmean_or_nan(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size else float("nan")


def nanstd_or_nan(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(finite.std(ddof=1)) if finite.size > 1 else float("nan")


def percentile_or_nan(values: pd.Series | np.ndarray, percentile: float) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.percentile(finite, percentile)) if finite.size else float("nan")


# -----------------------------------------------------------------------------
# Data preparation and fingerprint construction
# -----------------------------------------------------------------------------


def load_master() -> tuple[pd.DataFrame, int]:
    ensure_required_file(MASTER_PATH)
    frame = pd.read_parquet(MASTER_PATH)
    original_rows = len(frame)

    required = [ELIGIBILITY_COLUMN, COMPANY_COLUMN, YEAR_COLUMN, COMMUNITY_COLUMN]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError("Master dataset is missing required columns: " + ", ".join(missing))

    frame = frame.loc[frame[ELIGIBILITY_COLUMN].fillna(False).astype(bool)].copy()
    frame[YEAR_COLUMN] = pd.to_numeric(frame[YEAR_COLUMN], errors="coerce")
    frame = frame.dropna(subset=[COMPANY_COLUMN, YEAR_COLUMN, COMMUNITY_COLUMN]).copy()
    frame[YEAR_COLUMN] = frame[YEAR_COLUMN].astype(int)
    frame[COMPANY_COLUMN] = frame[COMPANY_COLUMN].astype(str)
    frame[COMMUNITY_COLUMN] = frame[COMMUNITY_COLUMN].astype(str)

    return frame, original_rows


def build_window_fingerprints(
    frame: pd.DataFrame,
    graph_features: list[str],
    communities: list[str],
) -> pd.DataFrame:
    records: list[dict] = []

    global_min_year = int(frame[YEAR_COLUMN].min())
    global_max_year = int(frame[YEAR_COLUMN].max())

    for company in sorted(frame[COMPANY_COLUMN].unique()):
        company_frame = frame.loc[frame[COMPANY_COLUMN] == company].copy()

        for window_size in WINDOW_SIZES:
            first_endpoint = global_min_year + window_size - 1
            for end_year in range(first_endpoint, global_max_year + 1):
                start_year = end_year - window_size + 1
                window = company_frame.loc[
                    company_frame[YEAR_COLUMN].between(start_year, end_year, inclusive="both")
                ].copy()

                n_songs = len(window)
                n_distinct_years = int(window[YEAR_COLUMN].nunique())
                if n_songs < MIN_SONGS_PER_WINDOW:
                    continue
                if n_distinct_years < MIN_DISTINCT_YEARS_PER_WINDOW:
                    continue

                record: dict[str, object] = {
                    "company": company,
                    "window_size": window_size,
                    "window_start_year": start_year,
                    "window_end_year": end_year,
                    "n_songs": n_songs,
                    "n_distinct_years": n_distinct_years,
                }

                for feature in graph_features:
                    values = pd.to_numeric(window[feature], errors="coerce")
                    record[f"graph_mean__{feature}"] = float(values.mean())

                counts = window[COMMUNITY_COLUMN].value_counts()
                for community in communities:
                    record[f"community_prop__{community}"] = float(counts.get(community, 0) / n_songs)

                records.append(record)

    if not records:
        raise RuntimeError(
            "No rolling-window fingerprints were created. Consider checking year coverage "
            "or lowering MIN_SONGS_PER_WINDOW."
        )

    fingerprints = pd.DataFrame.from_records(records)
    fingerprints = fingerprints.sort_values(
        ["company", "window_size", "window_end_year"]
    ).reset_index(drop=True)
    return fingerprints


# -----------------------------------------------------------------------------
# Forecasting models
# -----------------------------------------------------------------------------


def forecast_persistence(history_values: np.ndarray, history_years: np.ndarray) -> np.ndarray:
    del history_years
    return np.asarray(history_values[-1], dtype=float).copy()


def forecast_linear_trend(history_values: np.ndarray, history_years: np.ndarray) -> np.ndarray:
    values = np.asarray(history_values, dtype=float)
    years = np.asarray(history_years, dtype=float)
    target_year = years[-1] + 1.0
    prediction = np.full(values.shape[1], np.nan, dtype=float)

    for index in range(values.shape[1]):
        y = values[:, index]
        mask = np.isfinite(years) & np.isfinite(y)
        if mask.sum() < 2:
            finite = y[np.isfinite(y)]
            prediction[index] = finite[-1] if finite.size else np.nan
            continue
        x_valid = years[mask]
        y_valid = y[mask]
        if np.allclose(x_valid, x_valid[0]):
            prediction[index] = y_valid[-1]
            continue
        slope, intercept = np.polyfit(x_valid, y_valid, deg=1)
        prediction[index] = slope * target_year + intercept

    return prediction


def forecast_moving_average(history_values: np.ndarray, history_years: np.ndarray) -> np.ndarray:
    del history_years
    values = np.asarray(history_values, dtype=float)
    recent = values[-min(MOVING_AVERAGE_HISTORY, len(values)) :]
    return np.nanmean(recent, axis=0)


def model_prediction(
    model_name: str,
    history_values: np.ndarray,
    history_years: np.ndarray,
) -> np.ndarray:
    if model_name == "persistence":
        return forecast_persistence(history_values, history_years)
    if model_name == "linear_trend":
        return forecast_linear_trend(history_values, history_years)
    if model_name == "moving_average":
        return forecast_moving_average(history_values, history_years)
    raise ValueError(f"Unknown forecast model: {model_name}")


# -----------------------------------------------------------------------------
# Walk-forward evaluation
# -----------------------------------------------------------------------------


def run_walk_forward_forecasting(
    fingerprints: pd.DataFrame,
    graph_columns: list[str],
    community_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_records: list[dict] = []
    prediction_records: list[dict] = []

    groups = list(fingerprints.groupby(["company", "window_size"], sort=True))
    total_groups = len(groups)

    for group_index, ((company, window_size), group) in enumerate(groups, start=1):
        group = group.sort_values("window_end_year").reset_index(drop=True)
        print(
            f"[{group_index:02d}/{total_groups:02d}] {company}, {window_size}-year windows: "
            f"{len(group)} fingerprints",
            flush=True,
        )

        graph_matrix = group[graph_columns].to_numpy(dtype=float)
        community_matrix = group[community_columns].to_numpy(dtype=float)
        years = group["window_end_year"].to_numpy(dtype=int)

        # All models need at least two earlier fingerprints so that every model is
        # evaluated on exactly the same forecast targets.
        for target_index in range(2, len(group)):
            history_graph = graph_matrix[:target_index]
            history_community = community_matrix[:target_index]
            history_years = years[:target_index]

            actual_graph = graph_matrix[target_index]
            actual_community = normalise_probability_vector(community_matrix[target_index])
            target_year = int(years[target_index])

            for model_name in MODEL_NAMES:
                predicted_graph = model_prediction(model_name, history_graph, history_years)
                predicted_community_raw = model_prediction(
                    model_name, history_community, history_years
                )
                predicted_community = normalise_probability_vector(predicted_community_raw)

                graph_rmse = history_standardised_rmse(
                    actual_graph, predicted_graph, history_graph
                )
                standardised_actual_graph, standardised_predicted_graph = (
                    history_standardised_vectors(
                        actual_graph, predicted_graph, history_graph
                    )
                )
                graph_cosine = safe_cosine_similarity(
                    standardised_actual_graph, standardised_predicted_graph
                )
                community_js = safe_js_divergence(actual_community, predicted_community)
                community_cosine = safe_cosine_similarity(
                    actual_community, predicted_community
                )

                fold_id = (
                    f"{company}|w{window_size}|target{target_year}|{model_name}"
                )
                fold_records.append(
                    {
                        "fold_id": fold_id,
                        "company": company,
                        "window_size": int(window_size),
                        "target_window_start_year": int(
                            group.loc[target_index, "window_start_year"]
                        ),
                        "target_window_end_year": target_year,
                        "history_start_endpoint": int(history_years[0]),
                        "history_end_endpoint": int(history_years[-1]),
                        "n_history_fingerprints": int(target_index),
                        "n_target_songs": int(group.loc[target_index, "n_songs"]),
                        "model": model_name,
                        "graph_rmse": graph_rmse,
                        "graph_cosine_similarity": graph_cosine,
                        "community_js_divergence": community_js,
                        "community_cosine_similarity": community_cosine,
                    }
                )

                for feature_index, feature in enumerate(graph_columns):
                    prediction_records.append(
                        {
                            "fold_id": fold_id,
                            "company": company,
                            "window_size": int(window_size),
                            "target_window_end_year": target_year,
                            "model": model_name,
                            "fingerprint_component": "graph",
                            "feature": feature.replace("graph_mean__", ""),
                            "actual_value": float(actual_graph[feature_index]),
                            "predicted_value": float(predicted_graph[feature_index]),
                        }
                    )

                for feature_index, feature in enumerate(community_columns):
                    prediction_records.append(
                        {
                            "fold_id": fold_id,
                            "company": company,
                            "window_size": int(window_size),
                            "target_window_end_year": target_year,
                            "model": model_name,
                            "fingerprint_component": "community_distribution",
                            "feature": feature.replace("community_prop__", ""),
                            "actual_value": float(actual_community[feature_index]),
                            "predicted_value": float(predicted_community[feature_index]),
                        }
                    )

    if not fold_records:
        raise RuntimeError(
            "No valid walk-forward forecasts were produced. At least three valid rolling "
            "fingerprints per company/window-size combination are required."
        )

    return pd.DataFrame(fold_records), pd.DataFrame(prediction_records)


# -----------------------------------------------------------------------------
# Summaries
# -----------------------------------------------------------------------------


def interpretation_from_metrics(
    graph_cosine: float,
    community_js: float,
) -> str:
    if not np.isfinite(graph_cosine) or not np.isfinite(community_js):
        return "insufficient_metric_coverage"
    if graph_cosine >= 0.95 and community_js <= 0.03:
        return "strong_temporal_continuity"
    if graph_cosine >= 0.85 and community_js <= 0.08:
        return "moderate_temporal_continuity"
    if graph_cosine >= 0.70 and community_js <= 0.15:
        return "limited_temporal_continuity"
    return "weak_temporal_continuity"


def summarise_forecasts(folds: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["company", "window_size", "model"]
    records: list[dict] = []

    for keys, group in folds.groupby(group_columns, sort=True):
        company, window_size, model = keys
        record = {
            "company": company,
            "window_size": int(window_size),
            "model": model,
            "n_forecasts": int(len(group)),
        }
        for metric in [
            "graph_rmse",
            "graph_cosine_similarity",
            "community_js_divergence",
            "community_cosine_similarity",
        ]:
            record[f"{metric}_mean"] = nanmean_or_nan(group[metric])
            record[f"{metric}_std"] = nanstd_or_nan(group[metric])
            record[f"{metric}_ci_low"] = percentile_or_nan(group[metric], 2.5)
            record[f"{metric}_ci_high"] = percentile_or_nan(group[metric], 97.5)

        record["interpretation"] = interpretation_from_metrics(
            record["graph_cosine_similarity_mean"],
            record["community_js_divergence_mean"],
        )
        records.append(record)

    return pd.DataFrame(records).sort_values(group_columns).reset_index(drop=True)


def build_model_comparison(folds: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "graph_rmse": "lower",
        "graph_cosine_similarity": "higher",
        "community_js_divergence": "lower",
        "community_cosine_similarity": "higher",
    }
    records: list[dict] = []

    for window_size in sorted(folds["window_size"].unique()):
        subset = folds.loc[folds["window_size"] == window_size]
        for metric, direction in metrics.items():
            model_means = subset.groupby("model")[metric].mean()
            best_model = model_means.idxmin() if direction == "lower" else model_means.idxmax()
            persistence_value = float(model_means.get("persistence", np.nan))

            for model_name, score in model_means.items():
                score = float(score)
                absolute_improvement = float("nan")
                relative_improvement = float("nan")
                if np.isfinite(persistence_value):
                    absolute_improvement = (
                        persistence_value - score
                        if direction == "lower"
                        else score - persistence_value
                    )
                    denominator = abs(persistence_value)
                    if denominator > EPS:
                        relative_improvement = 100.0 * absolute_improvement / denominator

                records.append(
                    {
                        "window_size": int(window_size),
                        "metric": metric,
                        "direction": direction,
                        "model": model_name,
                        "mean_score": score,
                        "persistence_score": persistence_value,
                        "absolute_improvement_over_persistence": absolute_improvement,
                        "relative_improvement_over_persistence_pct": relative_improvement,
                        "best_model_for_metric": best_model,
                        "is_best_model": bool(model_name == best_model),
                    }
                )

    return pd.DataFrame(records).sort_values(
        ["window_size", "metric", "model"]
    ).reset_index(drop=True)


def build_statistical_comparisons(folds: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "graph_rmse": "lower",
        "graph_cosine_similarity": "higher",
        "community_js_divergence": "lower",
        "community_cosine_similarity": "higher",
    }
    records: list[dict] = []
    pairing_columns = ["company", "window_size", "target_window_end_year"]

    for comparator in ["moving_average", "linear_trend"]:
        for metric, direction in metrics.items():
            base = folds.loc[folds["model"] == "persistence", pairing_columns + [metric]].rename(
                columns={metric: "persistence_score"}
            )
            other = folds.loc[folds["model"] == comparator, pairing_columns + [metric]].rename(
                columns={metric: "comparator_score"}
            )
            paired = base.merge(other, on=pairing_columns, how="inner").dropna()

            if direction == "lower":
                advantage = paired["comparator_score"] - paired["persistence_score"]
            else:
                advantage = paired["persistence_score"] - paired["comparator_score"]

            nonzero = advantage[np.abs(advantage) > EPS]
            if len(nonzero) == 0:
                statistic = 0.0
                p_value = 1.0
            else:
                try:
                    statistic, p_value = wilcoxon(
                        nonzero,
                        alternative="greater",
                        zero_method="wilcox",
                        correction=False,
                    )
                    statistic = float(statistic)
                    p_value = float(p_value)
                except ValueError:
                    statistic = float("nan")
                    p_value = float("nan")

            records.append(
                {
                    "comparison": f"persistence_vs_{comparator}",
                    "comparator_model": comparator,
                    "metric": metric,
                    "direction": direction,
                    "n_pairs": int(len(paired)),
                    "persistence_mean": nanmean_or_nan(paired["persistence_score"]),
                    "comparator_mean": nanmean_or_nan(paired["comparator_score"]),
                    "mean_persistence_advantage": nanmean_or_nan(advantage),
                    "median_persistence_advantage": float(np.nanmedian(advantage)) if len(advantage) else float("nan"),
                    "wilcoxon_statistic": statistic,
                    "one_sided_p_value": p_value,
                    "significant_at_0_05": bool(np.isfinite(p_value) and p_value < 0.05),
                }
            )

    comparisons = pd.DataFrame(records)
    # Holm adjustment across the eight planned paired comparisons.
    valid = comparisons["one_sided_p_value"].notna()
    pvals = comparisons.loc[valid, "one_sided_p_value"].to_numpy(dtype=float)
    if len(pvals):
        order = np.argsort(pvals)
        adjusted = np.empty_like(pvals)
        running = 0.0
        m = len(pvals)
        for rank, idx in enumerate(order):
            candidate = min(1.0, (m - rank) * pvals[idx])
            running = max(running, candidate)
            adjusted[idx] = running
        comparisons.loc[valid, "holm_adjusted_p_value"] = adjusted
    else:
        comparisons["holm_adjusted_p_value"] = np.nan
    comparisons["significant_after_holm_0_05"] = (
        comparisons["holm_adjusted_p_value"] < 0.05
    )
    return comparisons


def build_company_summary(folds: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for (company, model_name), group in folds.groupby(["company", "model"], sort=True):
        record = {
            "company": company,
            "model": model_name,
            "n_forecasts": int(len(group)),
            "graph_rmse_mean": nanmean_or_nan(group["graph_rmse"]),
            "graph_cosine_similarity_mean": nanmean_or_nan(group["graph_cosine_similarity"]),
            "community_js_divergence_mean": nanmean_or_nan(group["community_js_divergence"]),
            "community_cosine_similarity_mean": nanmean_or_nan(group["community_cosine_similarity"]),
        }
        records.append(record)
    return pd.DataFrame(records).sort_values(["company", "model"]).reset_index(drop=True)


def build_validation_summary(folds: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for model_name in MODEL_NAMES:
        group = folds.loc[folds["model"] == model_name]
        record = {
            "analysis": "temporal_forecasting",
            "model": model_name,
            "n_forecasts": int(len(group)),
            "graph_rmse_mean": nanmean_or_nan(group["graph_rmse"]),
            "graph_cosine_similarity_mean": nanmean_or_nan(
                group["graph_cosine_similarity"]
            ),
            "community_js_divergence_mean": nanmean_or_nan(
                group["community_js_divergence"]
            ),
            "community_cosine_similarity_mean": nanmean_or_nan(
                group["community_cosine_similarity"]
            ),
        }
        records.append(record)

    summary = pd.DataFrame(records)
    summary["mean_rank"] = (
        summary[
            [
                "graph_rmse_mean",
                "community_js_divergence_mean",
            ]
        ]
        .rank(method="average", ascending=True)
        .mean(axis=1)
        + summary[
            [
                "graph_cosine_similarity_mean",
                "community_cosine_similarity_mean",
            ]
        ]
        .rank(method="average", ascending=False)
        .mean(axis=1)
    ) / 2.0
    summary["overall_rank"] = summary["mean_rank"].rank(method="min").astype(int)
    summary["interpretation"] = summary["overall_rank"].map(
        {1: "best_temporal_baseline", 2: "second_ranked_forecaster", 3: "third_ranked_forecaster"}
    ).fillna("ranked_forecaster")
    return summary.sort_values(["overall_rank", "model"]).reset_index(drop=True)


def build_feature_manifest(
    graph_features: list[str], communities: list[str]
) -> pd.DataFrame:
    records = [
        {
            "fingerprint_component": "graph",
            "feature": feature,
            "construction": "company-window arithmetic mean",
        }
        for feature in graph_features
    ]
    records.extend(
        {
            "fingerprint_component": "community_distribution",
            "feature": community,
            "construction": "company-window song proportion",
        }
        for community in communities
    )
    return pd.DataFrame(records)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    print_section(ANALYSIS_NAME)
    frame, original_rows = load_master()

    graph_features = existing_columns(frame, GRAPH_FEATURE_CANDIDATES)
    if not graph_features:
        raise KeyError(
            "None of the expected graph-derived features are present in the master table."
        )

    # Retain graph features with at least two finite values and non-zero variance.
    usable_graph_features: list[str] = []
    excluded_graph_features: list[str] = []
    for feature in graph_features:
        values = pd.to_numeric(frame[feature], errors="coerce")
        if values.notna().sum() < 2 or not np.isfinite(values.var(skipna=True)) or values.var(skipna=True) <= EPS:
            excluded_graph_features.append(feature)
        else:
            usable_graph_features.append(feature)
    graph_features = usable_graph_features

    communities = sorted(frame[COMMUNITY_COLUMN].dropna().astype(str).unique().tolist())

    print(f"Script version          : {SCRIPT_VERSION}")
    print(f"Project root            : {ROOT}")
    print(f"Master dataset          : {MASTER_PATH}")
    print(f"Output directory        : {OUTPUT_DIR}")
    print(f"Master rows loaded      : {original_rows}")
    print(f"Eligible rows retained  : {len(frame)}")
    print(f"Companies               : {frame[COMPANY_COLUMN].nunique()}")
    print(f"Year range              : {frame[YEAR_COLUMN].min()}-{frame[YEAR_COLUMN].max()}")
    print(f"Rolling windows         : {list(WINDOW_SIZES)}")
    print(f"Minimum songs/window    : {MIN_SONGS_PER_WINDOW}")
    print(f"Graph features          : {graph_features}")
    print(f"Communities             : {len(communities)}")
    if excluded_graph_features:
        print(f"Excluded graph features : {excluded_graph_features}")

    print("\nCompany-year coverage:")
    coverage = frame.groupby(COMPANY_COLUMN)[YEAR_COLUMN].agg(["min", "max", "nunique"])
    print(coverage.to_string())

    print_section("BUILDING ROLLING-WINDOW FINGERPRINTS")
    fingerprints = build_window_fingerprints(frame, graph_features, communities)
    graph_columns = [f"graph_mean__{feature}" for feature in graph_features]
    community_columns = [f"community_prop__{community}" for community in communities]

    print(f"Rolling fingerprints    : {len(fingerprints)}")
    print("\nFingerprints per company/window size:")
    print(
        fingerprints.groupby(["company", "window_size"]).size().rename("n").to_string()
    )

    print_section("RUNNING WALK-FORWARD ONE-STEP FORECASTS")
    folds, predictions = run_walk_forward_forecasting(
        fingerprints, graph_columns, community_columns
    )

    results = summarise_forecasts(folds)
    comparison = build_model_comparison(folds)
    validation_summary = build_validation_summary(folds)
    statistical_comparisons = build_statistical_comparisons(folds)
    company_summary = build_company_summary(folds)
    feature_manifest = build_feature_manifest(graph_features, communities)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_DIR / "forecast_results.csv", index=False)
    folds.to_csv(OUTPUT_DIR / "forecast_fold_results.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "forecast_predictions.csv", index=False)
    fingerprints.to_csv(OUTPUT_DIR / "forecast_window_fingerprints.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "forecast_model_comparison.csv", index=False)
    feature_manifest.to_csv(OUTPUT_DIR / "forecast_feature_manifest.csv", index=False)
    validation_summary.to_csv(
        OUTPUT_DIR / "forecast_validation_summary.csv", index=False
    )
    statistical_comparisons.to_csv(
        OUTPUT_DIR / "forecast_statistical_comparisons.csv", index=False
    )
    company_summary.to_csv(OUTPUT_DIR / "forecast_company_summary.csv", index=False)

    metadata = {
        "analysis_name": ANALYSIS_NAME,
        "script_version": SCRIPT_VERSION,
        "master_path": str(MASTER_PATH),
        "output_directory": str(OUTPUT_DIR),
        "eligible_rows": int(len(frame)),
        "companies": sorted(frame[COMPANY_COLUMN].unique().tolist()),
        "year_min": int(frame[YEAR_COLUMN].min()),
        "year_max": int(frame[YEAR_COLUMN].max()),
        "window_sizes": list(WINDOW_SIZES),
        "minimum_songs_per_window": MIN_SONGS_PER_WINDOW,
        "minimum_distinct_years_per_window": MIN_DISTINCT_YEARS_PER_WINDOW,
        "moving_average_history": MOVING_AVERAGE_HISTORY,
        "forecast_models": list(MODEL_NAMES),
        "graph_features": graph_features,
        "excluded_graph_features": excluded_graph_features,
        "communities": communities,
        "n_window_fingerprints": int(len(fingerprints)),
        "n_forecasts": int(len(folds)),
        "graph_cosine_standardisation": (
            "For every forecast origin, graph vectors are centred and scaled using "
            "only the available historical fingerprints before cosine similarity is calculated."
        ),
        "statistical_test": (
            "One-sided paired Wilcoxon signed-rank tests compare persistence with each "
            "alternative model across identical forecast targets; Holm adjustment is "
            "applied across the eight planned model-metric comparisons."
        ),
        "methodological_note": (
            "Strict expanding-history walk-forward evaluation; all models are "
            "evaluated on identical targets beginning with the third available "
            "rolling fingerprint for each company/window-size combination."
        ),
    }
    with (OUTPUT_DIR / "forecast_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print_section("FORECAST VALIDATION SUMMARY")
    display_columns = [
        "model",
        "n_forecasts",
        "graph_rmse_mean",
        "graph_cosine_similarity_mean",
        "community_js_divergence_mean",
        "community_cosine_similarity_mean",
        "overall_rank",
        "interpretation",
    ]
    print(validation_summary[display_columns].to_string(index=False))

    print_section("MODEL COMPARISON BY WINDOW SIZE")
    best_rows = comparison.loc[comparison["is_best_model"]].copy()
    print(
        best_rows[
            ["window_size", "metric", "best_model_for_metric", "mean_score"]
        ].to_string(index=False)
    )

    print_section("PAIRED STATISTICAL COMPARISONS")
    print(
        statistical_comparisons[
            [
                "comparison",
                "metric",
                "n_pairs",
                "mean_persistence_advantage",
                "one_sided_p_value",
                "holm_adjusted_p_value",
                "significant_after_holm_0_05",
            ]
        ].to_string(index=False)
    )

    print_section("COMPANY-WISE FORECAST SUMMARY")
    print(
        company_summary[
            [
                "company",
                "model",
                "n_forecasts",
                "graph_rmse_mean",
                "graph_cosine_similarity_mean",
                "community_js_divergence_mean",
                "community_cosine_similarity_mean",
            ]
        ].to_string(index=False)
    )

    print_section("OUTPUTS WRITTEN")
    for filename in [
        "forecast_results.csv",
        "forecast_fold_results.csv",
        "forecast_predictions.csv",
        "forecast_window_fingerprints.csv",
        "forecast_model_comparison.csv",
        "forecast_feature_manifest.csv",
        "forecast_validation_summary.csv",
        "forecast_statistical_comparisons.csv",
        "forecast_company_summary.csv",
        "forecast_metadata.json",
    ]:
        print(OUTPUT_DIR / filename)


if __name__ == "__main__":
    main()
