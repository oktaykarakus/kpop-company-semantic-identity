#!/usr/bin/env python3
"""
07a_company_prediction_v4.py

WP7a - Predictive validation of company semantic fingerprints with integrated feature-set ablation.

Scientific question
-------------------
Can song-level semantic graph features predict the entertainment company
that released a song?

The analysis uses only existing columns from:

    outputs/tables/company_master_analysis_v1.parquet

No file discovery is performed. The project root is fixed as:

    ROOT = Path(__file__).resolve().parents[2]

Validation designs
------------------
1. Repeated stratified cross-validation at song level.
2. Repeated stratified group cross-validation with artist as the group.

The artist-grouped analysis is a stricter test that prevents songs from the
same artist appearing in both training and test folds.

Models
------
- Multinomial Logistic Regression
- Random Forest

Outputs
-------
Written directly to the existing outputs/tables directory:

- prediction_results.csv
- prediction_fold_results.csv
- prediction_class_results.csv
- confusion_matrix.csv
- feature_importance.csv
- prediction_feature_manifest.csv
- prediction_validation_summary.csv
- prediction_ablation_summary.csv

No figures are generated in this script.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SCRIPT_VERSION = "2026-07-18-v4"
ANALYSIS_NAME = "WP7a : COMPANY PREDICTION"
RANDOM_SEED = 20260718
N_SPLITS = 5
N_REPEATS = 10
N_FEATURE_PERMUTATION_REPEATS = 5
N_NULL_PERMUTATIONS = 250
MAX_GROUP_SPLIT_ATTEMPTS = 500

ROOT = Path(__file__).resolve().parents[2]
MASTER_PATH = ROOT / "outputs" / "tables" / "company_master_analysis_v1.parquet"
OUTPUT_DIR = ROOT / "outputs" / "tables"

TARGET_COLUMN = "company"
ELIGIBILITY_COLUMN = "analysis_eligible"
GROUP_COLUMN_CANDIDATES = ["artist_normalised", "artist"]

# The script uses only columns already present in the master table. These are
# explicit semantic / graph feature candidates rather than a file-search or
# automatic inclusion of arbitrary numeric columns.
CATEGORICAL_FEATURE_CANDIDATES = [
    "community",
]

NUMERIC_FEATURE_CANDIDATES = [
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

LEAKAGE_EXACT_COLUMNS = {
    "company",
    "company_code",
    "company_id",
    "label",
    "target",
}

LEAKAGE_NAME_FRAGMENTS = (
    "company_mean",
    "company_profile",
    "company_centroid",
    "distance_to_company",
    "similarity_to_company",
    "predicted_company",
)


# -----------------------------------------------------------------------------
# Generic helpers
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


def select_existing_columns(frame: pd.DataFrame, candidates: Iterable[str]) -> list[str]:
    return [column for column in candidates if column in frame.columns]


def detect_group_column(frame: pd.DataFrame) -> str:
    for column in GROUP_COLUMN_CANDIDATES:
        if column in frame.columns:
            return column
    raise KeyError(
        "Artist-grouped validation requires one of these existing columns: "
        + ", ".join(GROUP_COLUMN_CANDIDATES)
    )


def remove_leakage_features(features: list[str]) -> list[str]:
    retained = []
    for feature in features:
        lowered = feature.lower()
        if lowered in LEAKAGE_EXACT_COLUMNS:
            continue
        if any(fragment in lowered for fragment in LEAKAGE_NAME_FRAGMENTS):
            continue
        retained.append(feature)
    return retained


def make_one_hot_encoder() -> OneHotEncoder:
    # sklearn renamed sparse -> sparse_output. This keeps the script compatible
    # with both older and newer environments.
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", make_one_hot_encoder()),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )

    if not transformers:
        raise ValueError("No usable graph-derived predictor columns were found.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_models(
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict[str, Pipeline]:
    logistic = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(numeric_features, categorical_features),
            ),
            (
                "classifier",
                LogisticRegression(
                    solver="lbfgs",
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    random_forest = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(numeric_features, categorical_features),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=3,
                    class_weight="balanced_subsample",
                    max_features="sqrt",
                    n_jobs=-1,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    dummy_most_frequent = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(numeric_features, categorical_features),
            ),
            ("classifier", DummyClassifier(strategy="most_frequent")),
        ]
    )

    dummy_stratified = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(numeric_features, categorical_features),
            ),
            (
                "classifier",
                DummyClassifier(strategy="stratified", random_state=RANDOM_SEED),
            ),
        ]
    )

    return {
        "dummy_most_frequent": dummy_most_frequent,
        "dummy_stratified": dummy_stratified,
        "logistic_regression": logistic,
        "random_forest": random_forest,
    }


def compute_global_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
        "precision_weighted": precision_score(
            y_true, y_pred, labels=labels, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_true, y_pred, labels=labels, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(
            y_true, y_pred, labels=labels, average="weighted", zero_division=0
        ),
    }


def interpret_predictive_strength(score: float) -> str:
    if not np.isfinite(score):
        return "not_available"
    if score < 0.30:
        return "limited"
    if score < 0.40:
        return "modest"
    if score < 0.55:
        return "moderate"
    if score < 0.70:
        return "strong"
    return "very_strong"


def interpret_statistical_support(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "not_available"
    if p_value < 0.01:
        return "strongly_supported"
    if p_value < 0.05:
        return "supported"
    return "not_supported"


# -----------------------------------------------------------------------------
# Cross-validation generators
# -----------------------------------------------------------------------------


def repeated_stratified_splits(
    features: pd.DataFrame,
    target: pd.Series,
):
    splitter = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_SEED,
    )
    for split_index, (train_index, test_index) in enumerate(
        splitter.split(features, target), start=1
    ):
        repeat_index = (split_index - 1) // N_SPLITS + 1
        fold_index = (split_index - 1) % N_SPLITS + 1
        yield repeat_index, fold_index, train_index, test_index


def _all_classes_present(
    target: pd.Series,
    train_index: np.ndarray,
    test_index: np.ndarray,
    labels: list[str],
) -> bool:
    required = set(labels)
    return (
        set(target.iloc[train_index].astype(str)) == required
        and set(target.iloc[test_index].astype(str)) == required
    )


def determine_grouped_split_count(
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    labels: list[str],
) -> int:
    """Find the largest grouped fold count yielding valid folds for every repeat."""
    maximum_possible = min(
        N_SPLITS,
        int(
            pd.DataFrame({"target": target, "group": groups})
            .groupby("target")["group"]
            .nunique()
            .min()
        ),
    )

    for candidate_splits in range(maximum_possible, 1, -1):
        feasible = True
        for repeat_index in range(1, N_REPEATS + 1):
            found = False
            for attempt in range(MAX_GROUP_SPLIT_ATTEMPTS):
                seed = RANDOM_SEED + (repeat_index - 1) * MAX_GROUP_SPLIT_ATTEMPTS + attempt
                splitter = StratifiedGroupKFold(
                    n_splits=candidate_splits,
                    shuffle=True,
                    random_state=seed,
                )
                candidate = list(splitter.split(features, target, groups))
                if all(
                    _all_classes_present(target, train_idx, test_idx, labels)
                    for train_idx, test_idx in candidate
                ):
                    found = True
                    break
            if not found:
                feasible = False
                break
        if feasible:
            return candidate_splits

    raise ValueError(
        "Could not construct artist-grouped cross-validation folds containing "
        "all companies in both training and test partitions."
    )


def repeated_artist_grouped_splits(
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    labels: list[str],
    grouped_n_splits: int,
):
    for repeat_index in range(1, N_REPEATS + 1):
        valid_splits = None
        selected_seed = None
        for attempt in range(MAX_GROUP_SPLIT_ATTEMPTS):
            seed = RANDOM_SEED + (repeat_index - 1) * MAX_GROUP_SPLIT_ATTEMPTS + attempt
            splitter = StratifiedGroupKFold(
                n_splits=grouped_n_splits,
                shuffle=True,
                random_state=seed,
            )
            candidate = list(splitter.split(features, target, groups))
            if all(
                _all_classes_present(target, train_idx, test_idx, labels)
                for train_idx, test_idx in candidate
            ):
                valid_splits = candidate
                selected_seed = seed
                break

        if valid_splits is None:
            raise RuntimeError(
                f"No valid grouped partition was found for repeat {repeat_index}."
            )

        for fold_index, (train_index, test_index) in enumerate(
            valid_splits, start=1
        ):
            yield (
                repeat_index,
                fold_index,
                train_index,
                test_index,
                selected_seed,
            )


# -----------------------------------------------------------------------------
# Core evaluation
# -----------------------------------------------------------------------------


def evaluate_validation_design(
    feature_set: str,
    validation_design: str,
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    labels: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    grouped_n_splits: int,
):
    models = build_models(numeric_features, categorical_features)

    fold_rows: list[dict] = []
    prediction_rows: list[dict] = []
    class_rows: list[dict] = []
    confusion_rows: list[dict] = []

    if validation_design == "repeated_stratified_song_level":
        cached_splits = [
            (repeat_index, fold_index, train_index, test_index, RANDOM_SEED)
            for repeat_index, fold_index, train_index, test_index
            in repeated_stratified_splits(features, target)
        ]
    elif validation_design == "repeated_stratified_artist_grouped":
        cached_splits = list(
            repeated_artist_grouped_splits(
                features, target, groups, labels, grouped_n_splits
            )
        )
    else:
        raise ValueError(f"Unknown validation design: {validation_design}")

    for model_name, pipeline in models.items():
        all_true: list[str] = []
        all_predicted: list[str] = []

        for split_number, split in enumerate(cached_splits, start=1):
            repeat_index, fold_index, train_index, test_index, split_seed = split
            x_train = features.iloc[train_index]
            x_test = features.iloc[test_index]
            y_train = target.iloc[train_index]
            y_test = target.iloc[test_index]

            # Give the stratified dummy a fold-specific seed while keeping all
            # substantive model settings fixed and reproducible.
            if model_name == "dummy_stratified":
                pipeline.set_params(classifier__random_state=split_seed + fold_index)

            pipeline.fit(x_train, y_train)
            y_pred = pipeline.predict(x_test)

            metrics = compute_global_metrics(
                y_test.to_numpy(), np.asarray(y_pred), labels
            )
            fold_rows.append(
                {
                    "feature_set": feature_set,
                    "validation_design": validation_design,
                    "model": model_name,
                    "repeat": repeat_index,
                    "fold": fold_index,
                    "split_number": split_number,
                    "split_seed": int(split_seed),
                    "n_train": len(train_index),
                    "n_test": len(test_index),
                    "n_train_artists": groups.iloc[train_index].nunique(),
                    "n_test_artists": groups.iloc[test_index].nunique(),
                    "n_train_classes": y_train.nunique(),
                    "n_test_classes": y_test.nunique(),
                    **metrics,
                }
            )

            for row_index, true_label, predicted_label in zip(
                test_index, y_test, y_pred
            ):
                prediction_rows.append(
                    {
                        "feature_set": feature_set,
                        "validation_design": validation_design,
                        "model": model_name,
                        "repeat": repeat_index,
                        "fold": fold_index,
                        "split_number": split_number,
                        "row_index": int(row_index),
                        "true_company": true_label,
                        "predicted_company": predicted_label,
                        "correct": int(true_label == predicted_label),
                    }
                )

            all_true.extend(y_test.astype(str).tolist())
            all_predicted.extend(pd.Series(y_pred).astype(str).tolist())


        all_true_array = np.asarray(all_true)
        all_predicted_array = np.asarray(all_predicted)

        precision, recall, f1, support = precision_recall_fscore_support(
            all_true_array,
            all_predicted_array,
            labels=labels,
            zero_division=0,
        )
        for company, p_value, r_value, f_value, n_value in zip(
            labels, precision, recall, f1, support
        ):
            class_rows.append(
                {
                    "feature_set": feature_set,
                    "validation_design": validation_design,
                    "model": model_name,
                    "company": company,
                    "precision": p_value,
                    "recall": r_value,
                    "f1": f_value,
                    "support_across_test_predictions": int(n_value),
                }
            )

        matrix = confusion_matrix(
            all_true_array,
            all_predicted_array,
            labels=labels,
        )
        for true_index, true_company in enumerate(labels):
            row_total = matrix[true_index].sum()
            for predicted_index, predicted_company in enumerate(labels):
                count = int(matrix[true_index, predicted_index])
                confusion_rows.append(
                    {
                        "feature_set": feature_set,
                        "validation_design": validation_design,
                        "model": model_name,
                        "true_company": true_company,
                        "predicted_company": predicted_company,
                        "count": count,
                        "row_normalised_proportion": (
                            count / row_total if row_total > 0 else np.nan
                        ),
                    }
                )

    return (
        pd.DataFrame(fold_rows),
        pd.DataFrame(prediction_rows),
        pd.DataFrame(class_rows),
        pd.DataFrame(confusion_rows),
    )



# -----------------------------------------------------------------------------
# Efficient feature importance
# -----------------------------------------------------------------------------


def compute_feature_importance_first_repeat(
    feature_set: str,
    validation_design: str,
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    labels: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    grouped_n_splits: int,
) -> pd.DataFrame:
    """Estimate permutation importance on only the first CV repeat.

    Running permutation importance inside every one of the 50 repeated-CV folds
    dominated runtime in v3. Five representative folds are sufficient for a
    stable descriptive ranking and keep prediction evaluation unchanged.
    """
    models = build_models(numeric_features, categorical_features)
    models = {k: v for k, v in models.items() if k in {"logistic_regression", "random_forest"}}

    if validation_design == "repeated_stratified_song_level":
        splits = [
            (r, f, tr, te, RANDOM_SEED)
            for r, f, tr, te in repeated_stratified_splits(features, target)
            if r == 1
        ]
    else:
        splits = [
            split for split in repeated_artist_grouped_splits(
                features, target, groups, labels, grouped_n_splits
            ) if split[0] == 1
        ]

    rows = []
    total = len(models) * len(splits)
    done = 0
    for model_name, pipeline in models.items():
        for repeat_index, fold_index, train_index, test_index, split_seed in splits:
            done += 1
            print(
                f"  Feature importance {feature_set} / {validation_design} / "
                f"{model_name}: fold {fold_index} ({done}/{total})",
                flush=True,
            )
            x_train = features.iloc[train_index]
            x_test = features.iloc[test_index]
            y_train = target.iloc[train_index]
            y_test = target.iloc[test_index]
            pipeline.fit(x_train, y_train)
            permutation = permutation_importance(
                pipeline,
                x_test,
                y_test,
                scoring="f1_macro",
                n_repeats=N_FEATURE_PERMUTATION_REPEATS,
                random_state=RANDOM_SEED + fold_index,
                n_jobs=1,
            )
            for feature_name, mean_value, std_value in zip(
                features.columns,
                permutation.importances_mean,
                permutation.importances_std,
            ):
                rows.append({
                    "feature_set": feature_set,
                    "validation_design": validation_design,
                    "model": model_name,
                    "repeat": repeat_index,
                    "fold": fold_index,
                    "feature": feature_name,
                    "importance_mean": mean_value,
                    "importance_std": std_value,
                    "importance_metric": "decrease_in_macro_f1",
                })
    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# Summary tables
# -----------------------------------------------------------------------------


def summarise_fold_results(fold_results: pd.DataFrame, n_classes: int) -> pd.DataFrame:
    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
    ]

    rows = []
    chance_level = 1.0 / n_classes

    for (feature_set, validation_design, model), subset in fold_results.groupby(
        ["feature_set", "validation_design", "model"], sort=True
    ):
        row = {
            "analysis": "Company prediction",
            "feature_set": feature_set,
            "validation_design": validation_design,
            "model": model,
            "n_folds": len(subset),
            "chance_accuracy": chance_level,
        }
        for metric in metric_columns:
            values = subset[metric].astype(float).dropna()
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std(ddof=1)
            row[f"{metric}_ci95_low"] = values.quantile(0.025)
            row[f"{metric}_ci95_high"] = values.quantile(0.975)

        row["accuracy_above_chance"] = row["accuracy_mean"] - chance_level
        row["balanced_accuracy_above_chance"] = (
            row["balanced_accuracy_mean"] - chance_level
        )
        row["predictive_strength"] = interpret_predictive_strength(
            row["f1_macro_mean"]
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["feature_set", "validation_design", "f1_macro_mean"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def run_prediction_permutation_tests(
    prediction_results: pd.DataFrame,
    labels: list[str],
) -> pd.DataFrame:
    """Fold-preserving permutation tests for substantive models only."""
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    substantive = prediction_results[
        prediction_results["model"].isin(["logistic_regression", "random_forest"])
    ]
    grouped_analyses = list(substantive.groupby(
        ["feature_set", "validation_design", "model"], sort=True
    ))

    for analysis_number, ((feature_set, validation_design, model), subset) in enumerate(
        grouped_analyses, start=1
    ):
        print(
            f"Permutation test {analysis_number}/{len(grouped_analyses)}: "
            f"{feature_set} / {validation_design} / {model}",
            flush=True,
        )
        folds = [fold for _, fold in subset.groupby(["repeat", "fold"], sort=True)]

        observed_macro_f1 = float(np.mean([
            f1_score(
                fold["true_company"].to_numpy(),
                fold["predicted_company"].to_numpy(),
                labels=labels,
                average="macro",
                zero_division=0,
            ) for fold in folds
        ]))
        observed_balanced_accuracy = float(np.mean([
            balanced_accuracy_score(
                fold["true_company"].to_numpy(),
                fold["predicted_company"].to_numpy(),
            ) for fold in folds
        ]))

        null_macro_f1 = np.empty(N_NULL_PERMUTATIONS, dtype=float)
        null_balanced_accuracy = np.empty(N_NULL_PERMUTATIONS, dtype=float)
        for permutation_index in range(N_NULL_PERMUTATIONS):
            fold_f1 = []
            fold_ba = []
            for fold in folds:
                shuffled_true = rng.permutation(fold["true_company"].to_numpy())
                predicted = fold["predicted_company"].to_numpy()
                fold_f1.append(f1_score(
                    shuffled_true, predicted, labels=labels,
                    average="macro", zero_division=0,
                ))
                fold_ba.append(balanced_accuracy_score(shuffled_true, predicted))
            null_macro_f1[permutation_index] = np.mean(fold_f1)
            null_balanced_accuracy[permutation_index] = np.mean(fold_ba)
            if (permutation_index + 1) % 25 == 0 or permutation_index == 0:
                print(
                    f"    {permutation_index + 1}/{N_NULL_PERMUTATIONS} permutations",
                    flush=True,
                )

        for metric_name, observed, null_values in [
            ("macro_f1", observed_macro_f1, null_macro_f1),
            ("balanced_accuracy", observed_balanced_accuracy, null_balanced_accuracy),
        ]:
            null_mean = float(null_values.mean())
            p_value = (1 + int(np.sum(null_values >= observed))) / (N_NULL_PERMUTATIONS + 1)
            rows.append({
                "feature_set": feature_set,
                "validation_design": validation_design,
                "model": model,
                "metric": metric_name,
                "observed_score": observed,
                "null_mean": null_mean,
                "null_std": float(null_values.std(ddof=1)),
                "null_ci95_low": float(np.quantile(null_values, 0.025)),
                "null_ci95_high": float(np.quantile(null_values, 0.975)),
                "score_above_null_mean": observed - null_mean,
                "relative_improvement_over_null_pct": (
                    100.0 * (observed - null_mean) / null_mean if null_mean else np.nan
                ),
                "empirical_p_value": p_value,
                "statistical_support": interpret_statistical_support(p_value),
                "n_permutations": N_NULL_PERMUTATIONS,
                "test_type": "fold_preserving_prediction_label_permutation",
            })
    return pd.DataFrame(rows)


def aggregate_feature_importance(importance_results: pd.DataFrame) -> pd.DataFrame:
    if importance_results.empty:
        return importance_results

    summary = (
        importance_results.groupby(
            ["feature_set", "validation_design", "model", "feature", "importance_metric"],
            as_index=False,
        )
        .agg(
            importance_mean=("importance_mean", "mean"),
            importance_std_across_folds=("importance_mean", "std"),
            mean_within_fold_permutation_std=("importance_std", "mean"),
            positive_importance_fold_fraction=(
                "importance_mean",
                lambda values: float((values > 0).mean()),
            ),
            n_folds=("importance_mean", "size"),
        )
    )
    summary["importance_rank"] = summary.groupby(
        ["feature_set", "validation_design", "model"]
    )["importance_mean"].rank(method="dense", ascending=False)

    return summary.sort_values(
        ["feature_set", "validation_design", "model", "importance_rank", "feature"]
    ).reset_index(drop=True)


def build_validation_summary(
    prediction_summary: pd.DataFrame,
    class_results: pd.DataFrame,
    feature_importance: pd.DataFrame,
    permutation_tests: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, result in prediction_summary.iterrows():
        feature_set = result["feature_set"]
        validation_design = result["validation_design"]
        model = result["model"]

        class_subset = class_results[
            (class_results["feature_set"] == feature_set)
            & (class_results["validation_design"] == validation_design)
            & (class_results["model"] == model)
        ]
        weakest_company = "not_available"
        weakest_company_f1 = np.nan
        if not class_subset.empty:
            weakest_row = class_subset.sort_values("f1").iloc[0]
            weakest_company = weakest_row["company"]
            weakest_company_f1 = weakest_row["f1"]

        importance_subset = feature_importance[
            (feature_importance["feature_set"] == feature_set)
            & (feature_importance["validation_design"] == validation_design)
            & (feature_importance["model"] == model)
        ] if not feature_importance.empty else pd.DataFrame()
        top_feature = "not_available"
        top_feature_importance = np.nan
        if not importance_subset.empty:
            top_row = importance_subset.sort_values("importance_mean", ascending=False).iloc[0]
            top_feature = top_row["feature"]
            top_feature_importance = top_row["importance_mean"]

        perm = permutation_tests[
            (permutation_tests["feature_set"] == feature_set)
            & (permutation_tests["validation_design"] == validation_design)
            & (permutation_tests["model"] == model)
            & (permutation_tests["metric"] == "macro_f1")
        ]
        null_mean = empirical_p = abs_gain = rel_gain = np.nan
        support = "not_available"
        if not perm.empty:
            prow = perm.iloc[0]
            null_mean = prow["null_mean"]
            empirical_p = prow["empirical_p_value"]
            abs_gain = prow["score_above_null_mean"]
            rel_gain = prow["relative_improvement_over_null_pct"]
            support = prow["statistical_support"]

        rows.append({
            "analysis": "Company prediction",
            "feature_set": feature_set,
            "validation_design": validation_design,
            "model": model,
            "primary_metric": "macro_f1",
            "primary_score": result["f1_macro_mean"],
            "primary_score_std": result["f1_macro_std"],
            "secondary_metric": "balanced_accuracy",
            "secondary_score": result["balanced_accuracy_mean"],
            "chance_level": result["chance_accuracy"],
            "score_above_chance": result["balanced_accuracy_above_chance"],
            "predictive_strength": result["predictive_strength"],
            "null_macro_f1_mean": null_mean,
            "absolute_macro_f1_gain_over_null": abs_gain,
            "relative_macro_f1_gain_over_null_pct": rel_gain,
            "empirical_p_value": empirical_p,
            "statistical_support": support,
            "weakest_company": weakest_company,
            "weakest_company_f1": weakest_company_f1,
            "top_feature": top_feature,
            "top_feature_importance": top_feature_importance,
        })

    return pd.DataFrame(rows)


def build_ablation_summary(validation_summary: pd.DataFrame) -> pd.DataFrame:
    substantive = validation_summary[
        validation_summary["model"].isin(["logistic_regression", "random_forest"])
    ].copy()
    combined = substantive[substantive["feature_set"] == "combined"][
        ["validation_design", "model", "primary_score", "secondary_score"]
    ].rename(columns={
        "primary_score": "combined_macro_f1",
        "secondary_score": "combined_balanced_accuracy",
    })
    result = substantive.merge(combined, on=["validation_design", "model"], how="left")
    result["macro_f1_difference_from_combined"] = (
        result["primary_score"] - result["combined_macro_f1"]
    )
    result["balanced_accuracy_difference_from_combined"] = (
        result["secondary_score"] - result["combined_balanced_accuracy"]
    )
    return result.sort_values(["validation_design", "model", "feature_set"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    print_section(ANALYSIS_NAME)
    print(f"Script version          : {SCRIPT_VERSION}")
    print(f"Project root            : {ROOT}")
    print(f"Master dataset          : {MASTER_PATH}")
    print(f"Output directory        : {OUTPUT_DIR}")

    ensure_required_file(MASTER_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    master = pd.read_parquet(MASTER_PATH)
    n_loaded = len(master)

    if ELIGIBILITY_COLUMN not in master.columns:
        raise KeyError(f"Required eligibility column '{ELIGIBILITY_COLUMN}' is absent.")
    if TARGET_COLUMN not in master.columns:
        raise KeyError(f"Required target column '{TARGET_COLUMN}' is absent.")

    master = master.loc[master[ELIGIBILITY_COLUMN].eq(True)].copy()
    master = master.loc[master[TARGET_COLUMN].notna()].copy()
    group_column = detect_group_column(master)
    master[group_column] = master[group_column].astype(str).str.strip()
    master[TARGET_COLUMN] = master[TARGET_COLUMN].astype(str).str.strip()

    categorical_features = remove_leakage_features(select_existing_columns(master, CATEGORICAL_FEATURE_CANDIDATES))
    numeric_features = remove_leakage_features(select_existing_columns(master, NUMERIC_FEATURE_CANDIDATES))
    retained_numeric, rejected_numeric = [], []
    for column in numeric_features:
        converted = pd.to_numeric(master[column], errors="coerce")
        if converted.notna().sum() >= N_SPLITS:
            master[column] = converted
            retained_numeric.append(column)
        else:
            rejected_numeric.append(column)
    numeric_features = retained_numeric

    if not categorical_features and not numeric_features:
        raise ValueError("No permitted semantic graph features were found.")

    all_selected = categorical_features + numeric_features
    analysis_frame = master[all_selected + [TARGET_COLUMN, group_column]].copy()
    valid_group = (analysis_frame[group_column].notna() & analysis_frame[group_column].astype(str).str.strip().ne("") & analysis_frame[group_column].astype(str).str.lower().ne("nan"))
    analysis_frame = analysis_frame.loc[valid_group].reset_index(drop=True)
    target = analysis_frame[TARGET_COLUMN]
    groups = analysis_frame[group_column]
    labels = sorted(target.unique().tolist())

    feature_sets = {
        "community_only": {"categorical": categorical_features, "numeric": []},
        "graph_only": {"categorical": [], "numeric": numeric_features},
        "combined": {"categorical": categorical_features, "numeric": numeric_features},
    }
    feature_sets = {name: spec for name, spec in feature_sets.items() if spec["categorical"] or spec["numeric"]}

    print(f"Master rows loaded      : {n_loaded}")
    print(f"Eligible rows retained  : {len(master)}")
    print(f"Prediction rows         : {len(analysis_frame)}")
    print(f"Companies               : {len(labels)} ({', '.join(labels)})")
    print(f"Artist group column     : {group_column}")
    print(f"Unique artist groups    : {groups.nunique()}")
    print(f"Feature sets            : {list(feature_sets)}")
    print(f"Categorical predictors  : {categorical_features}")
    print(f"Numeric predictors      : {numeric_features}")

    class_counts = target.value_counts().sort_index()
    artist_counts = analysis_frame[[TARGET_COLUMN, group_column]].drop_duplicates().groupby(TARGET_COLUMN)[group_column].nunique().sort_index()
    print("\nCompany song counts:")
    print(class_counts.to_string())
    print("\nUnique artist groups per company:")
    print(artist_counts.to_string())

    combined_features = analysis_frame[all_selected]
    grouped_n_splits = determine_grouped_split_count(combined_features, target, groups, labels)
    print(f"\nGrouped CV folds        : {grouped_n_splits}")

    all_fold_results, all_predictions, all_class_results, all_confusions = [], [], [], []
    validation_designs = ["repeated_stratified_song_level", "repeated_stratified_artist_grouped"]

    for feature_set, spec in feature_sets.items():
        selected = spec["categorical"] + spec["numeric"]
        features = analysis_frame[selected]
        for validation_design in validation_designs:
            print_section(f"RUNNING {feature_set.upper()} :: {validation_design.upper()}")
            outputs = evaluate_validation_design(
                feature_set=feature_set, validation_design=validation_design,
                features=features, target=target, groups=groups, labels=labels,
                numeric_features=spec["numeric"], categorical_features=spec["categorical"],
                grouped_n_splits=grouped_n_splits,
            )
            for collector, frame in zip([all_fold_results, all_predictions, all_class_results, all_confusions], outputs):
                collector.append(frame)

    fold_results = pd.concat(all_fold_results, ignore_index=True)
    prediction_results = pd.concat(all_predictions, ignore_index=True)
    class_results = pd.concat(all_class_results, ignore_index=True)
    confusion_results = pd.concat(all_confusions, ignore_index=True)
    results_summary = summarise_fold_results(fold_results, len(labels))

    # Save the expensive cross-validation outputs before secondary analyses.
    fold_results.to_csv(OUTPUT_DIR / "prediction_fold_results.csv", index=False)
    prediction_results.to_csv(OUTPUT_DIR / "prediction_out_of_fold_predictions.csv", index=False)
    class_results.to_csv(OUTPUT_DIR / "prediction_class_results.csv", index=False)
    confusion_results.to_csv(OUTPUT_DIR / "confusion_matrix.csv", index=False)
    results_summary.to_csv(OUTPUT_DIR / "prediction_results.csv", index=False)
    print("\nCore cross-validation outputs saved.", flush=True)

    print_section("EFFICIENT FEATURE IMPORTANCE (FIRST REPEAT ONLY)")
    importance_frames = []
    for feature_set, spec in feature_sets.items():
        selected = spec["categorical"] + spec["numeric"]
        features = analysis_frame[selected]
        for validation_design in validation_designs:
            importance_frames.append(compute_feature_importance_first_repeat(
                feature_set, validation_design, features, target, groups, labels,
                spec["numeric"], spec["categorical"], grouped_n_splits,
            ))
    raw_importance = pd.concat(importance_frames, ignore_index=True)
    feature_importance = aggregate_feature_importance(raw_importance)
    feature_importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    print_section("FOLD-PRESERVING PERMUTATION TESTS")
    permutation_tests = run_prediction_permutation_tests(prediction_results, labels)
    permutation_tests.to_csv(OUTPUT_DIR / "prediction_permutation_tests.csv", index=False)
    validation_summary = build_validation_summary(results_summary, class_results, feature_importance, permutation_tests)
    ablation_summary = build_ablation_summary(validation_summary)

    manifest = []
    for set_name, spec in feature_sets.items():
        for feature in spec["categorical"]:
            manifest.append({"feature_set": set_name, "feature": feature, "feature_type": "categorical", "included": True, "source": "company_master_analysis_v1.parquet"})
        for feature in spec["numeric"]:
            manifest.append({"feature_set": set_name, "feature": feature, "feature_type": "numeric", "included": True, "source": "company_master_analysis_v1.parquet"})
    feature_manifest = pd.DataFrame(manifest)

    results_summary.to_csv(OUTPUT_DIR / "prediction_results.csv", index=False)
    fold_results.to_csv(OUTPUT_DIR / "prediction_fold_results.csv", index=False)
    prediction_results.to_csv(OUTPUT_DIR / "prediction_out_of_fold_predictions.csv", index=False)
    class_results.to_csv(OUTPUT_DIR / "prediction_class_results.csv", index=False)
    confusion_results.to_csv(OUTPUT_DIR / "confusion_matrix.csv", index=False)
    feature_importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    feature_manifest.to_csv(OUTPUT_DIR / "prediction_feature_manifest.csv", index=False)
    validation_summary.to_csv(OUTPUT_DIR / "prediction_validation_summary.csv", index=False)
    permutation_tests.to_csv(OUTPUT_DIR / "prediction_permutation_tests.csv", index=False)
    ablation_summary.to_csv(OUTPUT_DIR / "prediction_ablation_summary.csv", index=False)
    artist_counts.rename("n_unique_artists").reset_index().to_csv(OUTPUT_DIR / "prediction_artist_counts.csv", index=False)

    metadata = {"script_version": SCRIPT_VERSION, "master_path": str(MASTER_PATH), "n_rows": int(len(analysis_frame)), "target": TARGET_COLUMN, "group_column": group_column, "companies": labels, "feature_sets": feature_sets, "validation_designs": validation_designs, "song_level_n_splits": N_SPLITS, "artist_grouped_n_splits": grouped_n_splits, "n_repeats": N_REPEATS, "n_null_permutations": N_NULL_PERMUTATIONS, "random_seed": RANDOM_SEED}
    (OUTPUT_DIR / "prediction_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print_section("PREDICTION RESULTS")
    cols = ["feature_set", "validation_design", "model", "balanced_accuracy_mean", "f1_macro_mean", "f1_macro_std", "predictive_strength"]
    print(results_summary[cols].to_string(index=False))
    print_section("PERMUTATION TESTS")
    cols = ["feature_set", "validation_design", "model", "metric", "observed_score", "null_mean", "score_above_null_mean", "relative_improvement_over_null_pct", "empirical_p_value", "statistical_support"]
    print(permutation_tests[cols].to_string(index=False))
    print_section("ABLATION SUMMARY")
    cols = ["feature_set", "validation_design", "model", "primary_score", "macro_f1_difference_from_combined", "predictive_strength", "statistical_support"]
    print(ablation_summary[cols].to_string(index=False))
    print_section("OUTPUT FILES")
    for filename in ["prediction_results.csv", "prediction_fold_results.csv", "prediction_out_of_fold_predictions.csv", "prediction_class_results.csv", "confusion_matrix.csv", "feature_importance.csv", "prediction_feature_manifest.csv", "prediction_validation_summary.csv", "prediction_permutation_tests.csv", "prediction_ablation_summary.csv", "prediction_artist_counts.csv", "prediction_metadata.json"]:
        print(OUTPUT_DIR / filename)


if __name__ == "__main__":
    main()
