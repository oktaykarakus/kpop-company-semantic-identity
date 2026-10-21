"""
WP6f: Community-detection and semantic-representation sensitivity.

This revision deliberately separates two questions that must not be conflated:

A. Same-representation community-detection sensitivity
   All alternative clusterings are fitted to one common song representation.
   A 15-cluster K-means solution using the primary seed is the internal
   reference. Changes in community count, algorithm, and K-means seed are then
   evaluated against that reference.

B. Alternative-representation sensitivity
   The original community labels stored in the master dataset are compared
   with the internally reconstructed representation. This is explicitly
   treated as a representation-transfer test, not as pure community-detection
   sensitivity.

Raw cosine distances from unrelated community-coordinate systems are not
subtracted from one another. Cross-solution conclusions use invariant relative
quantities: pairwise-distance correlation, distinctiveness-rank correlation,
rank-order retention, top/bottom grouping retention, ARI, and NMI.
"""

from __future__ import annotations

import ast
import json
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
        silhouette_score,
    )
    from sklearn.pipeline import FeatureUnion
except ImportError as exc:
    raise ImportError(
        "WP6f requires scikit-learn. Install it with: pip install scikit-learn"
    ) from exc


SCRIPT_VERSION = "2026-07-18-v4-separated-detection-representation"
ROOT = Path(__file__).resolve().parents[2]
MASTER_PATH = ROOT / "outputs" / "tables" / "company_master_analysis_v1.parquet"
OUTPUT_DIR = ROOT / "outputs" / "robustness" / "community_detection_sensitivity"

COMPANY_COLUMN = "company"
COMMUNITY_COLUMN = "community"
ELIGIBILITY_COLUMN = "analysis_eligible"
TEXT_COLUMN_CANDIDATES = ["lyrics", "lines", "text", "lyric_text"]
VECTOR_COLUMN_CANDIDATES = [
    "embedding", "embeddings", "sentence_embedding", "semantic_embedding",
    "lyrics_embedding", "song_embedding", "text_embedding",
    "sentence_transformer_embedding", "lyrics_vector", "semantic_vector",
    "features", "vector",
]
EMBEDDING_PREFIX_CANDIDATES = [
    "embedding_", "emb_", "sentence_embedding_", "semantic_embedding_",
    "dim_", "feature_",
]
IDENTIFIER_COLUMN_CANDIDATES = [
    "song_uid", "song_id", "track_id", "recording_id", "spotify_id",
    "row_id", "id",
]
EXTERNAL_EMBEDDING_CANDIDATES = [
    ROOT / "outputs" / "tables" / "song_embeddings.parquet",
    ROOT / "outputs" / "tables" / "semantic_embeddings.parquet",
    ROOT / "outputs" / "embeddings" / "song_embeddings.parquet",
    ROOT / "outputs" / "embeddings" / "semantic_embeddings.parquet",
    ROOT / "outputs" / "embeddings" / "song_embeddings.npy",
    ROOT / "outputs" / "embeddings" / "semantic_embeddings.npy",
    ROOT / "outputs" / "embeddings" / "song_embeddings.npz",
    ROOT / "outputs" / "embeddings" / "semantic_embeddings.npz",
]
DISCOVERY_ROOTS = [ROOT / "outputs", ROOT / "data", ROOT / "intermediate", ROOT / "processed"]
DISCOVERY_SUFFIXES = {".parquet", ".csv", ".npy", ".npz"}
DISCOVERY_KEYWORDS = ("embedding", "embeddings", "vector", "latent", "sentence")

COMMUNITY_COUNTS = [12, 15, 18]
KMEANS_SEEDS = [20260718, 20260719, 20260720, 20260721, 20260722]
PRIMARY_SEED = KMEANS_SEEDS[0]
INTERNAL_REFERENCE_K = 15
SILHOUETTE_SAMPLE_SIZE = 2000
EPSILON = 1e-12

TFIDF_MAX_FEATURES_WORD = 12000
TFIDF_MAX_FEATURES_CHAR = 18000
TFIDF_MIN_DF = 2
TEXT_SVD_COMPONENTS = 100
TEXT_SVD_RANDOM_STATE = PRIMARY_SEED

OUTPUT_PATHS = {
    "detection_solution_summary": OUTPUT_DIR / "community_detection_solution_summary.csv",
    "detection_company_rankings": OUTPUT_DIR / "community_detection_company_rankings.csv",
    "detection_company_sensitivity": OUTPUT_DIR / "community_detection_company_sensitivity.csv",
    "detection_global_summary": OUTPUT_DIR / "community_detection_global_summary.csv",
    "representation_summary": OUTPUT_DIR / "community_representation_sensitivity_summary.csv",
    "representation_company_rankings": OUTPUT_DIR / "community_representation_company_rankings.csv",
    "publication_summary": OUTPUT_DIR / "community_detection_publication_summary.csv",
}


def heading(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def safe_cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= EPSILON:
        return np.nan
    return float(1.0 - np.dot(a, b) / denom)


def safe_correlation(x: Iterable[float], y: Iterable[float]) -> float:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr, y_arr = x_arr[mask], y_arr[mask]
    if len(x_arr) < 2 or np.std(x_arr) <= EPSILON or np.std(y_arr) <= EPSILON:
        return np.nan
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def rank_correlation(reference: pd.DataFrame, alternative: pd.DataFrame) -> float:
    merged = reference[["company", "rank"]].merge(
        alternative[["company", "rank"]], on="company", suffixes=("_ref", "_alt")
    )
    return safe_correlation(merged["rank_ref"], merged["rank_alt"])


def parse_vector(value: object) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        arr = value.astype(float).ravel()
    elif isinstance(value, (list, tuple)):
        arr = np.asarray(value, dtype=float).ravel()
    elif isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            try:
                parsed = ast.literal_eval(value)
            except Exception:
                return None
        try:
            arr = np.asarray(parsed, dtype=float).ravel()
        except Exception:
            return None
    else:
        return None
    return arr if arr.size > 1 and np.all(np.isfinite(arr)) else None


def prepare_master(raw: pd.DataFrame) -> pd.DataFrame:
    required = {COMPANY_COLUMN, COMMUNITY_COLUMN, ELIGIBILITY_COLUMN}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise KeyError(f"Master dataset is missing required columns: {missing}")
    eligible = raw.loc[raw[ELIGIBILITY_COLUMN].fillna(False).astype(bool)].copy()
    eligible = eligible.dropna(subset=[COMPANY_COLUMN, COMMUNITY_COLUMN]).reset_index(drop=False)
    eligible = eligible.rename(columns={"index": "_master_original_index"})
    if eligible.empty:
        raise ValueError("No eligible rows remain after filtering.")
    return eligible


def normalise_embeddings(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] < 2:
        raise ValueError(f"Embedding matrix has invalid shape: {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Embedding matrix contains non-finite values.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms <= EPSILON] = 1.0
    return matrix / norms


def build_text_embeddings(master: pd.DataFrame) -> tuple[np.ndarray, str]:
    text_column = next((c for c in TEXT_COLUMN_CANDIDATES if c in master.columns), None)
    if text_column is None:
        raise FileNotFoundError("No usable embedding matrix and no text column were found.")
    texts = master[text_column].fillna("").astype(str)
    if (texts.str.strip() != "").sum() < 2:
        raise ValueError(f"Text column '{text_column}' does not contain enough usable documents.")
    union = FeatureUnion([
        ("word", TfidfVectorizer(
            lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
            min_df=TFIDF_MIN_DF, max_features=TFIDF_MAX_FEATURES_WORD,
            sublinear_tf=True, norm="l2",
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb", lowercase=True, strip_accents="unicode",
            ngram_range=(3, 5), min_df=TFIDF_MIN_DF,
            max_features=TFIDF_MAX_FEATURES_CHAR, sublinear_tf=True, norm="l2",
        )),
    ])
    sparse = union.fit_transform(texts)
    n_components = min(TEXT_SVD_COMPONENTS, sparse.shape[0] - 1, sparse.shape[1] - 1)
    if n_components < 2:
        raise ValueError("TF-IDF matrix is too small for a meaningful SVD representation.")
    svd = TruncatedSVD(n_components=n_components, random_state=TEXT_SVD_RANDOM_STATE)
    dense = svd.fit_transform(sparse)
    source = (
        f"generated from master '{text_column}' using word+character TF-IDF "
        f"and TruncatedSVD ({n_components} dimensions; cumulative explained "
        f"variance={svd.explained_variance_ratio_.sum():.4f})"
    )
    return dense, source


def load_external_table(path: Path, master: pd.DataFrame) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        matrix = np.load(path, allow_pickle=False)
        if matrix.shape[0] != len(master):
            raise ValueError(f"row count {matrix.shape[0]} != {len(master)}")
        return matrix
    if path.suffix.lower() == ".npz":
        archive = np.load(path, allow_pickle=False)
        keys = list(archive.keys())
        if not keys:
            raise ValueError("empty NPZ archive")
        matrix = archive[keys[0]]
        if matrix.shape[0] != len(master):
            raise ValueError(f"row count {matrix.shape[0]} != {len(master)}")
        return matrix
    table = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    vector_col = next((c for c in VECTOR_COLUMN_CANDIDATES if c in table.columns), None)
    numeric_cols = [
        c for c in table.columns
        if pd.api.types.is_numeric_dtype(table[c])
        and any(str(c).startswith(prefix) for prefix in EMBEDDING_PREFIX_CANDIDATES)
    ]
    shared_id = next((c for c in IDENTIFIER_COLUMN_CANDIDATES if c in table.columns and c in master.columns), None)
    if shared_id:
        aligned = master[[shared_id]].merge(table, on=shared_id, how="left", validate="one_to_one")
    elif len(table) == len(master):
        aligned = table.reset_index(drop=True)
    else:
        raise ValueError(f"row count {len(table)} != {len(master)} and no shared identifier")
    if vector_col:
        parsed = [parse_vector(v) for v in aligned[vector_col]]
        if any(v is None for v in parsed):
            raise ValueError(f"unparseable values in vector column '{vector_col}'")
        lengths = {len(v) for v in parsed if v is not None}
        if len(lengths) != 1:
            raise ValueError("inconsistent vector lengths")
        return np.vstack(parsed)
    if len(numeric_cols) >= 2:
        return aligned[numeric_cols].to_numpy(dtype=float)
    raise ValueError("no recognised vector representation")


def discover_candidates() -> list[Path]:
    candidates = list(EXTERNAL_EMBEDDING_CANDIDATES)
    for root in DISCOVERY_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in DISCOVERY_SUFFIXES:
                if any(k in path.name.lower() for k in DISCOVERY_KEYWORDS):
                    candidates.append(path)
    seen: set[Path] = set()
    return [p for p in candidates if not (p in seen or seen.add(p))]


def locate_common_representation(master: pd.DataFrame) -> tuple[np.ndarray, str, str]:
    for column in VECTOR_COLUMN_CANDIDATES:
        if column in master.columns:
            parsed = [parse_vector(v) for v in master[column]]
            if all(v is not None for v in parsed) and len({len(v) for v in parsed}) == 1:
                return np.vstack(parsed), f"master vector column '{column}'", "persisted_embedding"
    numeric_cols = [
        c for c in master.columns
        if pd.api.types.is_numeric_dtype(master[c])
        and any(str(c).startswith(prefix) for prefix in EMBEDDING_PREFIX_CANDIDATES)
    ]
    if len(numeric_cols) >= 2:
        return master[numeric_cols].to_numpy(float), f"master numeric columns ({len(numeric_cols)})", "persisted_embedding"
    for path in discover_candidates():
        if not path.exists() or path.resolve() == MASTER_PATH.resolve():
            continue
        try:
            matrix = load_external_table(path, master)
            return matrix, f"external file: {path}", "persisted_embedding"
        except Exception:
            continue
    matrix, source = build_text_embeddings(master)
    return matrix, source, "reconstructed_text_representation"


def cluster_labels(matrix: np.ndarray, algorithm: str, k: int, seed: int | None) -> np.ndarray:
    if algorithm == "kmeans":
        model = KMeans(n_clusters=k, random_state=seed, n_init=20, max_iter=500)
    elif algorithm == "agglomerative":
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    return np.asarray(model.fit_predict(matrix), dtype=int)


def build_profiles(companies: pd.Series, labels: np.ndarray, k: int) -> pd.DataFrame:
    frame = pd.DataFrame({"company": companies.to_numpy(), "label": labels})
    counts = pd.crosstab(frame["company"], frame["label"])
    counts = counts.reindex(columns=range(k), fill_value=0)
    profiles = counts.div(counts.sum(axis=1), axis=0)
    profiles.index.name = "company"
    return profiles


def build_pairwise(profiles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, b in combinations(sorted(profiles.index), 2):
        rows.append({
            "company_pair": f"{a}|{b}", "company_a": a, "company_b": b,
            "cosine_distance": safe_cosine_distance(profiles.loc[a].to_numpy(float), profiles.loc[b].to_numpy(float)),
        })
    return pd.DataFrame(rows).sort_values("company_pair").reset_index(drop=True)


def build_ranking(pairwise: pd.DataFrame) -> pd.DataFrame:
    companies = sorted(set(pairwise["company_a"]) | set(pairwise["company_b"]))
    rows = []
    for company in companies:
        values = pairwise.loc[(pairwise["company_a"] == company) | (pairwise["company_b"] == company), "cosine_distance"]
        rows.append({"company": company, "mean_pairwise_distance": float(values.mean())})
    ranking = pd.DataFrame(rows).sort_values(["mean_pairwise_distance", "company"], ascending=[False, True]).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    return ranking


def compare_solution(
    solution_id: str,
    analysis_block: str,
    algorithm: str,
    k: int,
    seed: int | None,
    labels: np.ndarray,
    matrix: np.ndarray,
    companies: pd.Series,
    ref_labels: np.ndarray,
    ref_pairs: pd.DataFrame,
    ref_ranking: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame]:
    profiles = build_profiles(companies, labels, k)
    pairs = build_pairwise(profiles)
    ranking = build_ranking(pairs)
    pair_merge = ref_pairs[["company_pair", "cosine_distance"]].merge(
        pairs[["company_pair", "cosine_distance"]], on="company_pair", suffixes=("_ref", "_alt")
    )
    ref_order = "|".join(ref_ranking.sort_values("rank")["company"])
    alt_order = "|".join(ranking.sort_values("rank")["company"])
    ref_top = set(ref_ranking.nsmallest(2, "rank")["company"])
    alt_top = set(ranking.nsmallest(2, "rank")["company"])
    merged_rank = ref_ranking[["company", "rank"]].merge(
        ranking[["company", "rank"]], on="company", suffixes=("_ref", "_alt")
    )
    max_disp = int((merged_rank["rank_alt"] - merged_rank["rank_ref"]).abs().max())
    sample_size = min(SILHOUETTE_SAMPLE_SIZE, len(matrix))
    silhouette = float(silhouette_score(matrix, labels, metric="cosine", sample_size=sample_size, random_state=PRIMARY_SEED))
    row = {
        "analysis_block": analysis_block,
        "solution_id": solution_id,
        "algorithm": algorithm,
        "n_communities_requested": k,
        "seed": seed,
        "adjusted_rand_index_vs_block_reference": float(adjusted_rand_score(ref_labels, labels)),
        "normalised_mutual_information_vs_block_reference": float(normalized_mutual_info_score(ref_labels, labels)),
        "cosine_silhouette": silhouette,
        "pairwise_distance_correlation": safe_correlation(pair_merge["cosine_distance_ref"], pair_merge["cosine_distance_alt"]),
        "distinctiveness_rank_correlation": rank_correlation(ref_ranking, ranking),
        "exact_rank_order_retained": ref_order == alt_order,
        "top_two_set_retained": ref_top == alt_top,
        "bottom_two_set_retained": ref_top == alt_top,
        "maximum_rank_displacement": max_disp,
        "reference_rank_order": ref_order,
        "alternative_rank_order": alt_order,
    }
    ranking = ranking.assign(solution_id=solution_id, analysis_block=analysis_block)
    return row, ranking


def classify_global(median_pair_corr: float, exact_prob: float, grouping_prob: float, max_disp: int) -> str:
    if median_pair_corr >= 0.90 and grouping_prob >= 0.75 and max_disp <= 1:
        return "highly_robust"
    if median_pair_corr >= 0.80 and grouping_prob >= 0.50 and max_disp <= 2:
        return "robust"
    if median_pair_corr >= 0.60 and exact_prob >= 0.25 and max_disp <= 2:
        return "moderately_robust"
    return "solution_sensitive"


def company_sensitivity(rankings: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ref = reference.set_index("company")
    for company, group in rankings.groupby("company", sort=True):
        reference_rank = int(ref.loc[company, "rank"])
        ranks = group["rank"].to_numpy(int)
        retention = float(np.mean(ranks == reference_rank))
        max_disp = int(np.max(np.abs(ranks - reference_rank)))
        if retention >= 0.80 and max_disp == 0:
            label = "highly_robust"
        elif retention >= 0.60 and max_disp <= 1:
            label = "robust"
        elif retention >= 0.40 and max_disp <= 2:
            label = "moderately_robust"
        else:
            label = "solution_sensitive"
        rows.append({
            "company": company,
            "reference_rank": reference_rank,
            "reference_rank_retention_probability": retention,
            "top_rank_probability": float(np.mean(ranks == 1)),
            "top_two_probability": float(np.mean(ranks <= 2)),
            "bottom_two_probability": float(np.mean(ranks >= 3)),
            "mean_rank_across_solutions": float(np.mean(ranks)),
            "minimum_rank_across_solutions": int(np.min(ranks)),
            "maximum_rank_across_solutions": int(np.max(ranks)),
            "maximum_absolute_rank_displacement": max_disp,
            "rank_robustness": label,
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Master dataset not found: {MASTER_PATH}")
    raw = pd.read_parquet(MASTER_PATH)
    master = prepare_master(raw)
    raw_matrix, source, representation_type = locate_common_representation(master)
    matrix = normalise_embeddings(raw_matrix)
    companies = master[COMPANY_COLUMN].astype(str)

    # Block A: internal same-representation reference.
    internal_ref_labels = cluster_labels(matrix, "kmeans", INTERNAL_REFERENCE_K, PRIMARY_SEED)
    internal_ref_profiles = build_profiles(companies, internal_ref_labels, INTERNAL_REFERENCE_K)
    internal_ref_pairs = build_pairwise(internal_ref_profiles)
    internal_ref_ranking = build_ranking(internal_ref_pairs)

    solution_specs: list[tuple[str, str, int, int | None]] = []
    for k in COMMUNITY_COUNTS:
        for seed in KMEANS_SEEDS:
            if k == INTERNAL_REFERENCE_K and seed == PRIMARY_SEED:
                continue
            solution_specs.append((f"kmeans_k{k}_seed{seed}", "kmeans", k, seed))
        solution_specs.append((f"agglomerative_k{k}", "agglomerative", k, None))

    detection_rows, detection_rankings = [], []
    for i, (solution_id, algorithm, k, seed) in enumerate(solution_specs, start=1):
        print(f"Processing detection solution {i:02d}/{len(solution_specs)}: {solution_id}")
        labels = cluster_labels(matrix, algorithm, k, seed)
        row, ranking = compare_solution(
            solution_id, "same_representation_detection", algorithm, k, seed,
            labels, matrix, companies, internal_ref_labels, internal_ref_pairs,
            internal_ref_ranking,
        )
        detection_rows.append(row)
        detection_rankings.append(ranking)
    detection_summary = pd.DataFrame(detection_rows)
    detection_ranking_table = pd.concat(detection_rankings, ignore_index=True)
    company_table = company_sensitivity(detection_ranking_table, internal_ref_ranking)

    median_pair_corr = float(detection_summary["pairwise_distance_correlation"].median())
    exact_prob = float(detection_summary["exact_rank_order_retained"].mean())
    top_prob = float(detection_summary["top_two_set_retained"].mean())
    max_disp = int(detection_summary["maximum_rank_displacement"].max())
    global_conclusion = classify_global(median_pair_corr, exact_prob, top_prob, max_disp)
    detection_global = pd.DataFrame([{
        "analysis": "Same-representation community-detection sensitivity",
        "representation_source": source,
        "representation_type": representation_type,
        "internal_reference_solution": f"kmeans_k{INTERNAL_REFERENCE_K}_seed{PRIMARY_SEED}",
        "n_alternative_solutions": len(detection_summary),
        "n_algorithms": int(detection_summary["algorithm"].nunique()),
        "n_community_counts": int(detection_summary["n_communities_requested"].nunique()),
        "median_ari": float(detection_summary["adjusted_rand_index_vs_block_reference"].median()),
        "median_nmi": float(detection_summary["normalised_mutual_information_vs_block_reference"].median()),
        "median_pairwise_distance_correlation": median_pair_corr,
        "minimum_pairwise_distance_correlation": float(detection_summary["pairwise_distance_correlation"].min()),
        "median_distinctiveness_rank_correlation": float(detection_summary["distinctiveness_rank_correlation"].median()),
        "exact_rank_order_probability": exact_prob,
        "top_two_set_probability": top_prob,
        "bottom_two_set_probability": float(detection_summary["bottom_two_set_retained"].mean()),
        "joint_grouping_probability": top_prob,
        "maximum_rank_displacement": max_disp,
        "global_conclusion": global_conclusion,
    }])

    # Block B: original stored labels versus reconstructed internal reference.
    original_codes, original_uniques = pd.factorize(master[COMMUNITY_COLUMN], sort=True)
    original_profiles = build_profiles(companies, original_codes, len(original_uniques))
    original_pairs = build_pairwise(original_profiles)
    original_ranking = build_ranking(original_pairs)
    pair_merge = original_pairs[["company_pair", "cosine_distance"]].merge(
        internal_ref_pairs[["company_pair", "cosine_distance"]], on="company_pair",
        suffixes=("_original", "_reconstructed"),
    )
    original_order = "|".join(original_ranking.sort_values("rank")["company"])
    reconstructed_order = "|".join(internal_ref_ranking.sort_values("rank")["company"])
    original_top = set(original_ranking.nsmallest(2, "rank")["company"])
    reconstructed_top = set(internal_ref_ranking.nsmallest(2, "rank")["company"])
    rank_merge = original_ranking[["company", "rank"]].merge(
        internal_ref_ranking[["company", "rank"]], on="company", suffixes=("_original", "_reconstructed")
    )
    representation_summary = pd.DataFrame([{
        "analysis": "Alternative semantic-representation sensitivity",
        "original_reference": "community labels stored in master dataset",
        "alternative_representation": source,
        "alternative_clustering": f"kmeans_k{INTERNAL_REFERENCE_K}_seed{PRIMARY_SEED}",
        "adjusted_rand_index": float(adjusted_rand_score(original_codes, internal_ref_labels)),
        "normalised_mutual_information": float(normalized_mutual_info_score(original_codes, internal_ref_labels)),
        "pairwise_distance_correlation": safe_correlation(pair_merge["cosine_distance_original"], pair_merge["cosine_distance_reconstructed"]),
        "distinctiveness_rank_correlation": rank_correlation(original_ranking, internal_ref_ranking),
        "exact_rank_order_retained": original_order == reconstructed_order,
        "top_two_set_retained": original_top == reconstructed_top,
        "bottom_two_set_retained": original_top == reconstructed_top,
        "maximum_rank_displacement": int((rank_merge["rank_reconstructed"] - rank_merge["rank_original"]).abs().max()),
        "original_rank_order": original_order,
        "alternative_rank_order": reconstructed_order,
        "interpretation_scope": (
            "This block changes the semantic representation and therefore is not "
            "used as the pure community-detection robustness conclusion."
        ),
    }])
    representation_rankings = pd.concat([
        original_ranking.assign(solution="original_master_communities"),
        internal_ref_ranking.assign(solution="reconstructed_internal_reference"),
    ], ignore_index=True)

    publication = pd.DataFrame([
        {
            "analysis": "Same-representation community-detection robustness",
            "finding_tested": "Company-level geometry and ordering under changes in k, seed, and algorithm",
            "primary_metric": "median pairwise-distance correlation",
            "primary_value": median_pair_corr,
            "secondary_metric": "exact rank-order retention probability",
            "secondary_value": exact_prob,
            "grouping_retention_probability": top_prob,
            "maximum_rank_displacement": max_disp,
            "robustness_conclusion": global_conclusion,
            "interpretation": (
                "This is the principal WP6f robustness result because all solutions "
                "share the same semantic representation."
            ),
        },
        {
            "analysis": "Alternative semantic-representation sensitivity",
            "finding_tested": "Transfer of company-level structure from original communities to reconstructed lyric representation",
            "primary_metric": "pairwise-distance correlation",
            "primary_value": representation_summary.iloc[0]["pairwise_distance_correlation"],
            "secondary_metric": "distinctiveness-rank correlation",
            "secondary_value": representation_summary.iloc[0]["distinctiveness_rank_correlation"],
            "grouping_retention_probability": float(representation_summary.iloc[0]["top_two_set_retained"]),
            "maximum_rank_displacement": int(representation_summary.iloc[0]["maximum_rank_displacement"]),
            "robustness_conclusion": "reported_separately",
            "interpretation": (
                "This supplementary block changes both representation and partition; "
                "it is not interpreted as pure community-detection sensitivity."
            ),
        },
    ])

    detection_summary.to_csv(OUTPUT_PATHS["detection_solution_summary"], index=False)
    detection_ranking_table.to_csv(OUTPUT_PATHS["detection_company_rankings"], index=False)
    company_table.to_csv(OUTPUT_PATHS["detection_company_sensitivity"], index=False)
    detection_global.to_csv(OUTPUT_PATHS["detection_global_summary"], index=False)
    representation_summary.to_csv(OUTPUT_PATHS["representation_summary"], index=False)
    representation_rankings.to_csv(OUTPUT_PATHS["representation_company_rankings"], index=False)
    publication.to_csv(OUTPUT_PATHS["publication_summary"], index=False)

    heading("WP6f : COMMUNITY-DETECTION AND REPRESENTATION SENSITIVITY")
    print(f"Script version          : {SCRIPT_VERSION}")
    print(f"Master dataset          : {MASTER_PATH}")
    print(f"Master rows loaded      : {len(raw)}")
    print(f"Eligible rows retained  : {len(master)}")
    print(f"Common representation   : {source}")
    print(f"Representation type     : {representation_type}")
    print(f"Embedding shape         : {matrix.shape}")
    print(f"Internal reference      : kmeans_k{INTERNAL_REFERENCE_K}_seed{PRIMARY_SEED}")
    print(f"Detection alternatives  : {len(detection_summary)}")

    heading("A. SAME-REPRESENTATION COMMUNITY-DETECTION SENSITIVITY")
    print(detection_global.to_string(index=False))
    heading("COMPANY RANKING SENSITIVITY WITHIN COMMON REPRESENTATION")
    print(company_table.to_string(index=False))
    heading("SOLUTION-LEVEL DETECTION RESULTS")
    print(detection_summary[[
        "solution_id", "algorithm", "n_communities_requested",
        "adjusted_rand_index_vs_block_reference",
        "normalised_mutual_information_vs_block_reference",
        "pairwise_distance_correlation", "distinctiveness_rank_correlation",
        "exact_rank_order_retained", "top_two_set_retained",
        "maximum_rank_displacement", "alternative_rank_order",
    ]].to_string(index=False))

    heading("B. ALTERNATIVE-REPRESENTATION SENSITIVITY")
    print(representation_summary.to_string(index=False))
    heading("PUBLICATION SUMMARY")
    print(publication.to_string(index=False))
    heading("OUTPUT FILES")
    for path in OUTPUT_PATHS.values():
        print(path)


if __name__ == "__main__":
    main()
