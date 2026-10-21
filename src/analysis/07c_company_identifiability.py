#!/usr/bin/env python3
"""
07c_company_identifiability.py

WP7c - Company semantic identifiability.

Scientific question
-------------------
How distinctive, recoverable, and temporally consistent are company-level
semantic fingerprints?

This script implements five complementary experiments:

1. Bootstrap identity stability
   Repeatedly sample 70% of each company's catalogue without replacement and
   compare the resulting fingerprint with every full-company reference.

2. Identification margin
   For every bootstrap sample, measure the distance to its own company reference
   relative to the nearest competing company reference.

3. Partial catalogue recovery
   Repeat identity recovery across catalogue fractions from 10% to 90%.

4. Nearest-company identity/confusion analysis
   Assign every bootstrap fingerprint to its nearest full-company reference and
   summarise recovery accuracy and confusion patterns.

5. Temporal identifiability
   Reuse WP7b rolling-window fingerprints and test whether each temporal
   fingerprint remains closest to the correct full-company reference.

Fingerprint design
------------------
A. Continuous graph fingerprint
   Company means of the graph/novelty variables already present in the master
   dataset. Feature values are standardised using the complete eligible song
   population before company means are calculated.

B. Semantic-community distribution
   Company proportions across the fixed semantic communities.

Distances and similarities
--------------------------
- graph_rms_distance: RMS distance between standardised graph means
- graph_cosine_similarity: cosine similarity between standardised graph means
- community_js_divergence: Jensen-Shannon divergence (base 2)
- community_js_similarity: 1 - Jensen-Shannon divergence
- community_cosine_similarity: cosine similarity between community proportions
- composite_distance: equal-weight average of graph RMS distance and
  Jensen-Shannon distance (the square root of JS divergence)

The primary nearest-company assignment uses composite_distance. Lower values are
better for distances; higher values are better for similarities.

Inputs
------
Required:
    outputs/tables/company_master_analysis_v1.parquet

Experiment 5 also expects the existing WP7b output:
    outputs/tables/forecast_window_fingerprints.csv

Outputs
-------
Written to outputs/tables:

- company_reference_fingerprints.csv
- bootstrap_similarity.csv
- identification_margin.csv
- catalogue_recovery_curves.csv
- nearest_company_results.csv
- nearest_company_confusion.csv
- identifiability_statistics.csv
- temporal_identifiability.csv
- temporal_identifiability_summary.csv
- temporal_identifiability_confusion.csv
- company_identifiability_summary.csv
- company_identifiability_metadata.json

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


SCRIPT_VERSION = "2026-07-19-v1"
ANALYSIS_NAME = "WP7c : COMPANY IDENTIFIABILITY"

ROOT = Path(__file__).resolve().parents[2]
MASTER_PATH = ROOT / "outputs" / "tables" / "company_master_analysis_v1.parquet"
TEMPORAL_FINGERPRINT_PATH = ROOT / "outputs" / "tables" / "forecast_window_fingerprints.csv"
OUTPUT_DIR = ROOT / "outputs" / "tables"

ELIGIBILITY_COLUMN = "analysis_eligible"
COMPANY_COLUMN = "company"
COMMUNITY_COLUMN = "community"
YEAR_COLUMN = "year"

RANDOM_SEED = 20260719
BOOTSTRAP_FRACTION = 0.70
N_BOOTSTRAP_REPEATS = 500
RECOVERY_FRACTIONS = tuple(np.round(np.arange(0.10, 1.00, 0.10), 2))
N_RECOVERY_REPEATS = 250
MIN_SAMPLE_SIZE = 3
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


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def print_section(title: str) -> None:
    line = "=" * 78
    print(f"\n{line}\n{title}\n{line}")


def ensure_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required {description} was not found:\n  {path}")


def existing_columns(frame: pd.DataFrame, candidates: Iterable[str]) -> list[str]:
    return [column for column in candidates if column in frame.columns]


def safe_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return float("nan")
    x = a[mask]
    y = b[mask]
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator <= EPS:
        return float("nan")
    return float(np.dot(x, y) / denominator)


def normalise_probability_vector(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    vector = np.where(np.isfinite(vector), vector, 0.0)
    vector = np.clip(vector, 0.0, None)
    total = float(vector.sum())
    if total <= EPS:
        return np.full(vector.shape, 1.0 / len(vector), dtype=float)
    return vector / total


def js_divergence(a: np.ndarray, b: np.ndarray) -> float:
    p = normalise_probability_vector(a)
    q = normalise_probability_vector(b)
    distance = float(jensenshannon(p, q, base=2.0))
    return distance * distance


def graph_rms_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))


def distance_bundle(
    graph_a: np.ndarray,
    community_a: np.ndarray,
    graph_b: np.ndarray,
    community_b: np.ndarray,
) -> dict[str, float]:
    graph_distance = graph_rms_distance(graph_a, graph_b)
    graph_cosine = safe_cosine_similarity(graph_a, graph_b)
    community_js = js_divergence(community_a, community_b)
    community_js_distance = math.sqrt(max(community_js, 0.0))
    community_cosine = safe_cosine_similarity(community_a, community_b)

    if np.isfinite(graph_distance) and np.isfinite(community_js_distance):
        composite = 0.5 * graph_distance + 0.5 * community_js_distance
    else:
        composite = float("nan")

    return {
        "graph_rms_distance": graph_distance,
        "graph_cosine_similarity": graph_cosine,
        "community_js_divergence": community_js,
        "community_js_similarity": 1.0 - community_js,
        "community_cosine_similarity": community_cosine,
        "composite_distance": composite,
    }


def cliff_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta using pairwise comparisons; positive means x > y."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")

    y_sorted = np.sort(y)
    greater = 0
    less = 0
    for value in x:
        less += int(np.searchsorted(y_sorted, value, side="left"))
        greater += int(y.size - np.searchsorted(y_sorted, value, side="right"))
    return float((less - greater) / (x.size * y.size))


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(p.shape, np.nan, dtype=float)
    valid_indices = np.flatnonzero(np.isfinite(p))
    if valid_indices.size == 0:
        return adjusted

    order = valid_indices[np.argsort(p[valid_indices])]
    m = len(order)
    running_max = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * p[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted


# -----------------------------------------------------------------------------
# Data and fingerprint construction
# -----------------------------------------------------------------------------


def load_master() -> tuple[pd.DataFrame, int, list[str], list[str]]:
    ensure_file(MASTER_PATH, "master dataset")
    frame = pd.read_parquet(MASTER_PATH)
    original_rows = len(frame)

    required = [ELIGIBILITY_COLUMN, COMPANY_COLUMN, COMMUNITY_COLUMN]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError("Master dataset is missing required columns: " + ", ".join(missing))

    frame = frame.loc[frame[ELIGIBILITY_COLUMN].fillna(False).astype(bool)].copy()
    frame = frame.dropna(subset=[COMPANY_COLUMN, COMMUNITY_COLUMN]).copy()
    frame[COMPANY_COLUMN] = frame[COMPANY_COLUMN].astype(str)
    frame[COMMUNITY_COLUMN] = frame[COMMUNITY_COLUMN].astype(str)

    graph_features = existing_columns(frame, GRAPH_FEATURE_CANDIDATES)
    if not graph_features:
        raise RuntimeError("No graph features were found in the master dataset.")

    for feature in graph_features:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")

    # Retain rows with at least one finite graph value. Missing feature values are
    # later imputed using the global eligible-song mean for that feature.
    finite_graph = np.isfinite(frame[graph_features].to_numpy(dtype=float)).any(axis=1)
    frame = frame.loc[finite_graph].copy().reset_index(drop=True)

    communities = sorted(frame[COMMUNITY_COLUMN].unique().tolist())
    return frame, original_rows, graph_features, communities


def fit_global_graph_standardiser(
    frame: pd.DataFrame,
    graph_features: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    values = frame[graph_features].to_numpy(dtype=float)
    means = np.nanmean(values, axis=0)
    scales = np.nanstd(values, axis=0, ddof=0)
    means = np.where(np.isfinite(means), means, 0.0)
    scales = np.where(np.isfinite(scales) & (scales > EPS), scales, 1.0)
    return means, scales


def prepare_standardised_song_matrix(
    frame: pd.DataFrame,
    graph_features: list[str],
    means: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    values = frame[graph_features].to_numpy(dtype=float)
    values = np.where(np.isfinite(values), values, means)
    return (values - means) / scales


def fingerprint_from_indices(
    frame: pd.DataFrame,
    standardised_graph: np.ndarray,
    indices: np.ndarray,
    communities: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(indices, dtype=int)
    graph_vector = np.nanmean(standardised_graph[indices], axis=0)
    community_values = frame.iloc[indices][COMMUNITY_COLUMN]
    counts = community_values.value_counts()
    community_vector = np.array(
        [float(counts.get(community, 0)) for community in communities], dtype=float
    )
    community_vector = normalise_probability_vector(community_vector)
    return graph_vector, community_vector


def build_reference_fingerprints(
    frame: pd.DataFrame,
    standardised_graph: np.ndarray,
    graph_features: list[str],
    communities: list[str],
) -> tuple[dict[str, dict[str, np.ndarray]], pd.DataFrame]:
    references: dict[str, dict[str, np.ndarray]] = {}
    records: list[dict] = []

    for company in sorted(frame[COMPANY_COLUMN].unique()):
        indices = np.flatnonzero(frame[COMPANY_COLUMN].to_numpy() == company)
        graph_vector, community_vector = fingerprint_from_indices(
            frame, standardised_graph, indices, communities
        )
        references[company] = {
            "graph": graph_vector,
            "community": community_vector,
        }

        record: dict[str, object] = {
            "company": company,
            "n_songs": int(len(indices)),
        }
        for index, feature in enumerate(graph_features):
            record[f"graph_mean_standardised__{feature}"] = float(graph_vector[index])
        for index, community in enumerate(communities):
            record[f"community_prop__{community}"] = float(community_vector[index])
        records.append(record)

    return references, pd.DataFrame(records)


def compare_to_references(
    source_company: str,
    sample_graph: np.ndarray,
    sample_community: np.ndarray,
    references: dict[str, dict[str, np.ndarray]],
) -> list[dict]:
    comparisons: list[dict] = []
    for reference_company, reference in references.items():
        metrics = distance_bundle(
            sample_graph,
            sample_community,
            reference["graph"],
            reference["community"],
        )
        comparisons.append(
            {
                "source_company": source_company,
                "reference_company": reference_company,
                "is_own_company": bool(source_company == reference_company),
                **metrics,
            }
        )
    return comparisons


# -----------------------------------------------------------------------------
# Experiments 1, 2 and 4: bootstrap identity stability and nearest identity
# -----------------------------------------------------------------------------


def run_bootstrap_identifiability(
    frame: pd.DataFrame,
    standardised_graph: np.ndarray,
    communities: list[str],
    references: dict[str, dict[str, np.ndarray]],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    similarity_records: list[dict] = []
    margin_records: list[dict] = []
    nearest_records: list[dict] = []

    companies = sorted(references)
    company_values = frame[COMPANY_COLUMN].to_numpy()

    for company_index, company in enumerate(companies, start=1):
        company_indices = np.flatnonzero(company_values == company)
        sample_size = max(MIN_SAMPLE_SIZE, int(round(BOOTSTRAP_FRACTION * len(company_indices))))
        sample_size = min(sample_size, len(company_indices))
        print(
            f"[{company_index:02d}/{len(companies):02d}] {company}: "
            f"{N_BOOTSTRAP_REPEATS} samples of {sample_size}/{len(company_indices)} songs",
            flush=True,
        )

        for repeat in range(1, N_BOOTSTRAP_REPEATS + 1):
            sampled = rng.choice(company_indices, size=sample_size, replace=False)
            sample_graph, sample_community = fingerprint_from_indices(
                frame, standardised_graph, sampled, communities
            )
            comparisons = compare_to_references(
                company, sample_graph, sample_community, references
            )

            for comparison in comparisons:
                similarity_records.append(
                    {
                        "experiment": "bootstrap_70pct",
                        "source_company": company,
                        "repeat": repeat,
                        "sample_fraction": BOOTSTRAP_FRACTION,
                        "sample_size": sample_size,
                        **{k: v for k, v in comparison.items() if k != "source_company"},
                    }
                )

            ordered = sorted(comparisons, key=lambda item: item["composite_distance"])
            nearest = ordered[0]
            own = next(item for item in comparisons if item["is_own_company"])
            nearest_other = next(item for item in ordered if not item["is_own_company"])
            margin = nearest_other["composite_distance"] - own["composite_distance"]

            margin_records.append(
                {
                    "source_company": company,
                    "repeat": repeat,
                    "sample_fraction": BOOTSTRAP_FRACTION,
                    "sample_size": sample_size,
                    "own_composite_distance": own["composite_distance"],
                    "nearest_other_company": nearest_other["reference_company"],
                    "nearest_other_composite_distance": nearest_other["composite_distance"],
                    "identification_margin": margin,
                    "positive_margin": bool(margin > 0),
                }
            )

            nearest_records.append(
                {
                    "source_company": company,
                    "repeat": repeat,
                    "sample_fraction": BOOTSTRAP_FRACTION,
                    "sample_size": sample_size,
                    "nearest_company": nearest["reference_company"],
                    "nearest_composite_distance": nearest["composite_distance"],
                    "correct_identification": bool(nearest["reference_company"] == company),
                    "identification_margin": margin,
                }
            )

    return (
        pd.DataFrame(similarity_records),
        pd.DataFrame(margin_records),
        pd.DataFrame(nearest_records),
    )


# -----------------------------------------------------------------------------
# Experiment 3: partial catalogue recovery
# -----------------------------------------------------------------------------


def run_catalogue_recovery(
    frame: pd.DataFrame,
    standardised_graph: np.ndarray,
    communities: list[str],
    references: dict[str, dict[str, np.ndarray]],
    rng: np.random.Generator,
) -> pd.DataFrame:
    records: list[dict] = []
    companies = sorted(references)
    company_values = frame[COMPANY_COLUMN].to_numpy()

    total_jobs = len(companies) * len(RECOVERY_FRACTIONS)
    job = 0
    for company in companies:
        company_indices = np.flatnonzero(company_values == company)
        for fraction in RECOVERY_FRACTIONS:
            job += 1
            sample_size = max(MIN_SAMPLE_SIZE, int(round(fraction * len(company_indices))))
            sample_size = min(sample_size, len(company_indices))
            print(
                f"[{job:02d}/{total_jobs:02d}] {company}, {fraction:.0%}: "
                f"{N_RECOVERY_REPEATS} samples of {sample_size} songs",
                flush=True,
            )

            for repeat in range(1, N_RECOVERY_REPEATS + 1):
                sampled = rng.choice(company_indices, size=sample_size, replace=False)
                sample_graph, sample_community = fingerprint_from_indices(
                    frame, standardised_graph, sampled, communities
                )
                comparisons = compare_to_references(
                    company, sample_graph, sample_community, references
                )
                ordered = sorted(comparisons, key=lambda item: item["composite_distance"])
                nearest = ordered[0]
                own = next(item for item in comparisons if item["is_own_company"])
                nearest_other = next(item for item in ordered if not item["is_own_company"])
                margin = nearest_other["composite_distance"] - own["composite_distance"]

                records.append(
                    {
                        "company": company,
                        "sample_fraction": float(fraction),
                        "sample_size": sample_size,
                        "repeat": repeat,
                        "own_graph_rms_distance": own["graph_rms_distance"],
                        "own_graph_cosine_similarity": own["graph_cosine_similarity"],
                        "own_community_js_divergence": own["community_js_divergence"],
                        "own_community_js_similarity": own["community_js_similarity"],
                        "own_community_cosine_similarity": own[
                            "community_cosine_similarity"
                        ],
                        "own_composite_distance": own["composite_distance"],
                        "nearest_company": nearest["reference_company"],
                        "correct_identification": bool(
                            nearest["reference_company"] == company
                        ),
                        "nearest_other_company": nearest_other["reference_company"],
                        "nearest_other_composite_distance": nearest_other[
                            "composite_distance"
                        ],
                        "identification_margin": margin,
                    }
                )

    return pd.DataFrame(records)


# -----------------------------------------------------------------------------
# Experiment 5: temporal identifiability
# -----------------------------------------------------------------------------


def run_temporal_identifiability(
    graph_features: list[str],
    communities: list[str],
    graph_means: np.ndarray,
    graph_scales: np.ndarray,
    references: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    ensure_file(TEMPORAL_FINGERPRINT_PATH, "WP7b rolling-window fingerprint output")
    temporal = pd.read_csv(TEMPORAL_FINGERPRINT_PATH)

    required_meta = [
        "company",
        "window_size",
        "window_start_year",
        "window_end_year",
        "n_songs",
    ]
    missing_meta = [column for column in required_meta if column not in temporal.columns]
    if missing_meta:
        raise KeyError(
            "WP7b temporal fingerprint file is missing columns: " + ", ".join(missing_meta)
        )

    graph_columns = [f"graph_mean__{feature}" for feature in graph_features]
    community_columns = [f"community_prop__{community}" for community in communities]
    missing_graph = [column for column in graph_columns if column not in temporal.columns]
    missing_community = [column for column in community_columns if column not in temporal.columns]
    if missing_graph or missing_community:
        details = []
        if missing_graph:
            details.append("graph=" + ", ".join(missing_graph))
        if missing_community:
            details.append("community=" + ", ".join(missing_community))
        raise KeyError("WP7b fingerprint columns do not match WP7c: " + "; ".join(details))

    records: list[dict] = []
    for row_index, row in temporal.iterrows():
        source_company = str(row["company"])
        raw_graph = row[graph_columns].to_numpy(dtype=float)
        raw_graph = np.where(np.isfinite(raw_graph), raw_graph, graph_means)
        standardised_graph = (raw_graph - graph_means) / graph_scales
        community_vector = normalise_probability_vector(
            row[community_columns].to_numpy(dtype=float)
        )

        comparisons = compare_to_references(
            source_company, standardised_graph, community_vector, references
        )
        ordered = sorted(comparisons, key=lambda item: item["composite_distance"])
        nearest = ordered[0]
        own = next(item for item in comparisons if item["is_own_company"])
        nearest_other = next(item for item in ordered if not item["is_own_company"])
        margin = nearest_other["composite_distance"] - own["composite_distance"]

        records.append(
            {
                "temporal_fingerprint_id": int(row_index),
                "source_company": source_company,
                "window_size": int(row["window_size"]),
                "window_start_year": int(row["window_start_year"]),
                "window_end_year": int(row["window_end_year"]),
                "n_songs": int(row["n_songs"]),
                "nearest_company": nearest["reference_company"],
                "correct_identification": bool(
                    nearest["reference_company"] == source_company
                ),
                "own_composite_distance": own["composite_distance"],
                "nearest_other_company": nearest_other["reference_company"],
                "nearest_other_composite_distance": nearest_other[
                    "composite_distance"
                ],
                "identification_margin": margin,
                "own_graph_rms_distance": own["graph_rms_distance"],
                "own_graph_cosine_similarity": own["graph_cosine_similarity"],
                "own_community_js_divergence": own["community_js_divergence"],
                "own_community_cosine_similarity": own[
                    "community_cosine_similarity"
                ],
            }
        )

    return pd.DataFrame(records)


# -----------------------------------------------------------------------------
# Statistical and descriptive summaries
# -----------------------------------------------------------------------------


def build_identifiability_statistics(similarities: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    metrics = {
        "graph_cosine_similarity": "higher",
        "community_js_similarity": "higher",
        "community_cosine_similarity": "higher",
        "graph_rms_distance": "lower",
        "community_js_divergence": "lower",
        "composite_distance": "lower",
    }

    for company in sorted(similarities["source_company"].unique()):
        subset = similarities.loc[similarities["source_company"] == company].copy()
        own = subset.loc[subset["is_own_company"]].sort_values("repeat")

        for metric, direction in metrics.items():
            # For the paired test, compare each own score with the closest competing
            # reference under the same metric and repeat.
            between = subset.loc[~subset["is_own_company"]].copy()
            if direction == "higher":
                competitor = between.groupby("repeat")[metric].max()
                difference = own.set_index("repeat")[metric] - competitor
                alternative = "greater"
            else:
                competitor = between.groupby("repeat")[metric].min()
                difference = competitor - own.set_index("repeat")[metric]
                alternative = "less"

            aligned = pd.concat(
                [own.set_index("repeat")[metric].rename("own"), competitor.rename("other")],
                axis=1,
                join="inner",
            ).dropna()

            if len(aligned) >= 2 and not np.allclose(aligned["own"], aligned["other"]):
                try:
                    statistic, p_value = wilcoxon(
                        aligned["own"],
                        aligned["other"],
                        alternative=alternative,
                        zero_method="wilcox",
                    )
                except ValueError:
                    statistic, p_value = float("nan"), float("nan")
            else:
                statistic, p_value = float("nan"), float("nan")

            own_values = own[metric].to_numpy(dtype=float)
            all_other_values = between[metric].to_numpy(dtype=float)
            if direction == "higher":
                delta = cliff_delta(own_values, all_other_values)
            else:
                # Reverse order so positive delta always indicates stronger
                # identifiability for the own-company comparison.
                delta = cliff_delta(all_other_values, own_values)

            records.append(
                {
                    "company": company,
                    "metric": metric,
                    "direction": direction,
                    "n_pairs": int(len(aligned)),
                    "own_mean": float(np.nanmean(own_values)),
                    "nearest_competitor_mean": float(np.nanmean(competitor.to_numpy(dtype=float))),
                    "mean_identifiability_advantage": float(np.nanmean(difference.to_numpy(dtype=float))),
                    "wilcoxon_statistic": float(statistic),
                    "one_sided_p_value": float(p_value),
                    "cliffs_delta": delta,
                }
            )

    result = pd.DataFrame(records)
    result["holm_adjusted_p_value"] = holm_adjust(result["one_sided_p_value"].to_numpy())
    result["significant_after_holm_0_05"] = result["holm_adjusted_p_value"] < 0.05
    return result.sort_values(["company", "metric"]).reset_index(drop=True)


def build_confusion_table(
    frame: pd.DataFrame,
    actual_column: str,
    predicted_column: str,
    count_name: str,
) -> pd.DataFrame:
    counts = pd.crosstab(frame[actual_column], frame[predicted_column], dropna=False)
    long = counts.stack(future_stack=True).rename(count_name).reset_index()
    totals = long.groupby(actual_column)[count_name].transform("sum")
    long["row_proportion"] = np.where(totals > 0, long[count_name] / totals, np.nan)
    return long


def summarise_recovery(recovery: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for (company, fraction), group in recovery.groupby(
        ["company", "sample_fraction"], sort=True
    ):
        records.append(
            {
                "company": company,
                "sample_fraction": float(fraction),
                "sample_size": int(group["sample_size"].iloc[0]),
                "n_repeats": int(len(group)),
                "identification_accuracy": float(group["correct_identification"].mean()),
                "positive_margin_rate": float((group["identification_margin"] > 0).mean()),
                "identification_margin_mean": float(group["identification_margin"].mean()),
                "identification_margin_median": float(group["identification_margin"].median()),
                "own_composite_distance_mean": float(group["own_composite_distance"].mean()),
                "own_graph_cosine_similarity_mean": float(
                    group["own_graph_cosine_similarity"].mean()
                ),
                "own_community_js_divergence_mean": float(
                    group["own_community_js_divergence"].mean()
                ),
                "own_community_cosine_similarity_mean": float(
                    group["own_community_cosine_similarity"].mean()
                ),
            }
        )
    return pd.DataFrame(records)


def summarise_temporal(temporal: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    grouping_options = [
        ("overall", ["window_size"]),
        ("company", ["source_company", "window_size"]),
    ]
    for summary_level, columns in grouping_options:
        for keys, group in temporal.groupby(columns, sort=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            record: dict[str, object] = {"summary_level": summary_level}
            for column, value in zip(columns, keys):
                record[column] = value
            record.update(
                {
                    "n_temporal_fingerprints": int(len(group)),
                    "identification_accuracy": float(group["correct_identification"].mean()),
                    "positive_margin_rate": float((group["identification_margin"] > 0).mean()),
                    "identification_margin_mean": float(group["identification_margin"].mean()),
                    "own_composite_distance_mean": float(group["own_composite_distance"].mean()),
                    "own_graph_cosine_similarity_mean": float(
                        group["own_graph_cosine_similarity"].mean()
                    ),
                    "own_community_js_divergence_mean": float(
                        group["own_community_js_divergence"].mean()
                    ),
                    "own_community_cosine_similarity_mean": float(
                        group["own_community_cosine_similarity"].mean()
                    ),
                }
            )
            records.append(record)
    return pd.DataFrame(records)


def build_company_summary(
    margins: pd.DataFrame,
    nearest: pd.DataFrame,
    recovery_summary: pd.DataFrame,
    temporal: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict] = []
    for company in sorted(margins["source_company"].unique()):
        margin_group = margins.loc[margins["source_company"] == company]
        nearest_group = nearest.loc[nearest["source_company"] == company]
        company_recovery = recovery_summary.loc[recovery_summary["company"] == company]
        temporal_group = temporal.loc[temporal["source_company"] == company]

        stable_fraction = float("nan")
        eligible = company_recovery.loc[
            company_recovery["identification_accuracy"] >= 0.95
        ]
        if not eligible.empty:
            stable_fraction = float(eligible["sample_fraction"].min())

        records.append(
            {
                "company": company,
                "bootstrap_identification_accuracy": float(
                    nearest_group["correct_identification"].mean()
                ),
                "bootstrap_positive_margin_rate": float(
                    (margin_group["identification_margin"] > 0).mean()
                ),
                "bootstrap_margin_mean": float(
                    margin_group["identification_margin"].mean()
                ),
                "minimum_fraction_for_95pct_recovery": stable_fraction,
                "temporal_identification_accuracy": float(
                    temporal_group["correct_identification"].mean()
                ),
                "temporal_positive_margin_rate": float(
                    (temporal_group["identification_margin"] > 0).mean()
                ),
                "temporal_margin_mean": float(
                    temporal_group["identification_margin"].mean()
                ),
                "n_temporal_fingerprints": int(len(temporal_group)),
            }
        )

    return pd.DataFrame(records)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    print_section(ANALYSIS_NAME)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frame, original_rows, graph_features, communities = load_master()
    graph_means, graph_scales = fit_global_graph_standardiser(frame, graph_features)
    standardised_graph = prepare_standardised_song_matrix(
        frame, graph_features, graph_means, graph_scales
    )
    references, reference_table = build_reference_fingerprints(
        frame, standardised_graph, graph_features, communities
    )

    print(f"Script version          : {SCRIPT_VERSION}")
    print(f"Project root            : {ROOT}")
    print(f"Master dataset          : {MASTER_PATH}")
    print(f"Temporal fingerprints   : {TEMPORAL_FINGERPRINT_PATH}")
    print(f"Output directory        : {OUTPUT_DIR}")
    print(f"Master rows loaded      : {original_rows}")
    print(f"Eligible rows retained  : {len(frame)}")
    print(f"Companies               : {len(references)} ({', '.join(sorted(references))})")
    print(f"Graph features          : {graph_features}")
    print(f"Communities             : {len(communities)}")
    print(f"Bootstrap fraction      : {BOOTSTRAP_FRACTION:.0%}")
    print(f"Bootstrap repeats       : {N_BOOTSTRAP_REPEATS}")
    print(f"Recovery fractions      : {list(RECOVERY_FRACTIONS)}")
    print(f"Recovery repeats        : {N_RECOVERY_REPEATS}")
    print("\nCompany catalogue sizes:")
    print(frame.groupby(COMPANY_COLUMN).size().rename("n_songs").to_string())

    rng = np.random.default_rng(RANDOM_SEED)

    print_section("EXPERIMENTS 1, 2 AND 4: BOOTSTRAP IDENTITY STABILITY")
    bootstrap_similarity, margins, nearest = run_bootstrap_identifiability(
        frame, standardised_graph, communities, references, rng
    )

    print_section("EXPERIMENT 3: PARTIAL CATALOGUE RECOVERY")
    recovery_raw = run_catalogue_recovery(
        frame, standardised_graph, communities, references, rng
    )
    recovery_summary = summarise_recovery(recovery_raw)

    print_section("EXPERIMENT 5: TEMPORAL IDENTIFIABILITY")
    temporal = run_temporal_identifiability(
        graph_features,
        communities,
        graph_means,
        graph_scales,
        references,
    )

    statistics = build_identifiability_statistics(bootstrap_similarity)
    nearest_confusion = build_confusion_table(
        nearest,
        actual_column="source_company",
        predicted_column="nearest_company",
        count_name="n_bootstrap_samples",
    )
    temporal_summary = summarise_temporal(temporal)
    temporal_confusion = build_confusion_table(
        temporal,
        actual_column="source_company",
        predicted_column="nearest_company",
        count_name="n_temporal_fingerprints",
    )
    company_summary = build_company_summary(
        margins, nearest, recovery_summary, temporal
    )

    # Persist outputs.
    reference_table.to_csv(
        OUTPUT_DIR / "company_reference_fingerprints.csv", index=False
    )
    bootstrap_similarity.to_csv(
        OUTPUT_DIR / "bootstrap_similarity.csv", index=False
    )
    margins.to_csv(OUTPUT_DIR / "identification_margin.csv", index=False)
    recovery_summary.to_csv(
        OUTPUT_DIR / "catalogue_recovery_curves.csv", index=False
    )
    recovery_raw.to_csv(
        OUTPUT_DIR / "catalogue_recovery_samples.csv", index=False
    )
    nearest.to_csv(OUTPUT_DIR / "nearest_company_results.csv", index=False)
    nearest_confusion.to_csv(
        OUTPUT_DIR / "nearest_company_confusion.csv", index=False
    )
    statistics.to_csv(
        OUTPUT_DIR / "identifiability_statistics.csv", index=False
    )
    temporal.to_csv(OUTPUT_DIR / "temporal_identifiability.csv", index=False)
    temporal_summary.to_csv(
        OUTPUT_DIR / "temporal_identifiability_summary.csv", index=False
    )
    temporal_confusion.to_csv(
        OUTPUT_DIR / "temporal_identifiability_confusion.csv", index=False
    )
    company_summary.to_csv(
        OUTPUT_DIR / "company_identifiability_summary.csv", index=False
    )

    metadata = {
        "script_version": SCRIPT_VERSION,
        "analysis_name": ANALYSIS_NAME,
        "master_path": str(MASTER_PATH),
        "temporal_fingerprint_path": str(TEMPORAL_FINGERPRINT_PATH),
        "n_master_rows_loaded": int(original_rows),
        "n_eligible_rows": int(len(frame)),
        "companies": sorted(references),
        "graph_features": graph_features,
        "communities": communities,
        "random_seed": RANDOM_SEED,
        "bootstrap_fraction": BOOTSTRAP_FRACTION,
        "n_bootstrap_repeats": N_BOOTSTRAP_REPEATS,
        "recovery_fractions": list(RECOVERY_FRACTIONS),
        "n_recovery_repeats": N_RECOVERY_REPEATS,
        "primary_assignment_metric": "composite_distance",
        "composite_distance_definition": (
            "0.5 * graph_rms_distance + 0.5 * sqrt(community_js_divergence)"
        ),
        "graph_standardisation": (
            "Song-level graph features standardised using the complete eligible-song "
            "population; company and sample fingerprints are means in that fixed space."
        ),
        "temporal_experiment": (
            "Each WP7b rolling-window fingerprint is compared with every full-company "
            "reference using the same fixed graph standardisation and community basis."
        ),
    }
    with (OUTPUT_DIR / "company_identifiability_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2)

    print_section("BOOTSTRAP IDENTIFICATION SUMMARY")
    bootstrap_display = (
        nearest.groupby("source_company")
        .agg(
            n_samples=("correct_identification", "size"),
            identification_accuracy=("correct_identification", "mean"),
            mean_margin=("identification_margin", "mean"),
        )
        .reset_index()
    )
    print(bootstrap_display.to_string(index=False))

    print_section("PARTIAL CATALOGUE RECOVERY SUMMARY")
    display_fractions = recovery_summary.loc[
        recovery_summary["sample_fraction"].isin([0.1, 0.3, 0.5, 0.7, 0.9])
    ]
    print(
        display_fractions[
            [
                "company",
                "sample_fraction",
                "sample_size",
                "identification_accuracy",
                "identification_margin_mean",
            ]
        ].to_string(index=False)
    )

    print_section("TEMPORAL IDENTIFIABILITY SUMMARY")
    temporal_display = (
        temporal.groupby("source_company")
        .agg(
            n_temporal_fingerprints=("correct_identification", "size"),
            identification_accuracy=("correct_identification", "mean"),
            mean_margin=("identification_margin", "mean"),
        )
        .reset_index()
    )
    print(temporal_display.to_string(index=False))

    print_section("COMPANY IDENTIFIABILITY SUMMARY")
    print(company_summary.to_string(index=False))

    print_section("OUTPUTS WRITTEN")
    for filename in [
        "company_reference_fingerprints.csv",
        "bootstrap_similarity.csv",
        "identification_margin.csv",
        "catalogue_recovery_curves.csv",
        "catalogue_recovery_samples.csv",
        "nearest_company_results.csv",
        "nearest_company_confusion.csv",
        "identifiability_statistics.csv",
        "temporal_identifiability.csv",
        "temporal_identifiability_summary.csv",
        "temporal_identifiability_confusion.csv",
        "company_identifiability_summary.csv",
        "company_identifiability_metadata.json",
    ]:
        print(f"- {OUTPUT_DIR / filename}")


if __name__ == "__main__":
    main()
