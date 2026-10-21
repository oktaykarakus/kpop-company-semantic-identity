"""WP6d: Balanced company-sampling robustness analysis.

Equalises every company catalogue to the size of the smallest eligible
company, then rebuilds company fingerprints, pairwise distances and
company distinctiveness rankings over repeated samples.

Profile robustness and rank robustness are deliberately reported as
separate conclusions. The smallest company defines the balancing threshold
and is therefore retained in full in every replicate.
"""

from __future__ import annotations

from itertools import combinations
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd


# ======================================================================
# CONFIGURATION
# ======================================================================

ROOT = Path(__file__).resolve().parents[2]
MASTER_PATH = ROOT / "outputs" / "tables" / "company_master_analysis_v1.parquet"
OUTPUT_DIR = ROOT / "outputs" / "robustness" / "balanced_company_sampling"

BALANCED_PROFILES_PATH = OUTPUT_DIR / "balanced_company_profiles.csv"
BALANCED_PAIRS_PATH = OUTPUT_DIR / "balanced_company_pairs.csv"
BALANCED_RANKINGS_PATH = OUTPUT_DIR / "balanced_company_rankings.csv"
PROFILE_CHANGES_PATH = OUTPUT_DIR / "balanced_company_profile_changes.csv"
PROFILE_SUMMARY_PATH = OUTPUT_DIR / "balanced_company_profile_summary.csv"
PAIR_SUMMARY_PATH = OUTPUT_DIR / "balanced_company_pair_summary.csv"
RANKING_SUMMARY_PATH = OUTPUT_DIR / "balanced_company_ranking_summary.csv"
RANKING_STABILITY_PATH = OUTPUT_DIR / "balanced_company_ranking_stability.csv"
PUBLICATION_SUMMARY_PATH = OUTPUT_DIR / "balanced_company_sampling_summary.csv"

N_REPLICATES = 2000
RANDOM_SEED = 20260718
CONFIDENCE_LEVEL = 0.95
EPSILON = 1e-12

COMPANY_COLUMN = "company"
COMMUNITY_COLUMN = "community"
ELIGIBILITY_COLUMN = "analysis_eligible"

COMPANY_METRIC_COLUMNS = [
    "degree",
    "weighted_degree",
    "bridge_affinity_score",
    "local_novelty",
    "centroid_novelty",
]


# ======================================================================
# UTILITIES
# ======================================================================

def print_heading(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def community_column_name(community: int) -> str:
    return f"community_{community}"


def confidence_limits() -> tuple[float, float]:
    alpha = 1.0 - CONFIDENCE_LEVEL
    return alpha / 2.0, 1.0 - alpha / 2.0


def cosine_distance(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    vector_a = np.asarray(vector_a, dtype=float)
    vector_b = np.asarray(vector_b, dtype=float)
    norm_a = float(np.linalg.norm(vector_a))
    norm_b = float(np.linalg.norm(vector_b))

    if norm_a <= EPSILON or norm_b <= EPSILON:
        raise ValueError("Cosine distance is undefined for a zero vector.")

    similarity = float(np.dot(vector_a, vector_b) / (norm_a * norm_b))
    return float(1.0 - np.clip(similarity, -1.0, 1.0))


def summarise_numeric_values(values: pd.Series, observed_value: float) -> dict:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)

    if len(numeric) == 0:
        return {
            "observed_value": float(observed_value),
            "replicate_mean": np.nan,
            "replicate_median": np.nan,
            "replicate_std": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "absolute_ci_width": np.nan,
            "sampling_bias": np.nan,
            "observed_within_ci": False,
        }

    lower_q, upper_q = confidence_limits()
    ci_lower = float(np.quantile(numeric, lower_q))
    ci_upper = float(np.quantile(numeric, upper_q))
    replicate_mean = float(np.mean(numeric))

    return {
        "observed_value": float(observed_value),
        "replicate_mean": replicate_mean,
        "replicate_median": float(np.median(numeric)),
        "replicate_std": float(np.std(numeric, ddof=1)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "absolute_ci_width": float(ci_upper - ci_lower),
        "sampling_bias": float(replicate_mean - observed_value),
        "observed_within_ci": bool(ci_lower <= observed_value <= ci_upper),
    }


# ======================================================================
# DATA PREPARATION
# ======================================================================

def validate_master_dataset(dataframe: pd.DataFrame) -> None:
    required = {
        COMPANY_COLUMN,
        COMMUNITY_COLUMN,
        ELIGIBILITY_COLUMN,
        *COMPANY_METRIC_COLUMNS,
    }
    missing = sorted(required.difference(dataframe.columns))

    if missing:
        raise ValueError(
            "The master dataset is missing required columns: " + ", ".join(missing)
        )
    if dataframe.empty:
        raise ValueError("The master dataset is empty.")


def prepare_master_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    working = dataframe.copy()
    eligibility = working[ELIGIBILITY_COLUMN]

    if pd.api.types.is_bool_dtype(eligibility):
        mask = eligibility
    else:
        mask = (
            eligibility.astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1", "yes"})
        )

    working = working.loc[mask].copy().reset_index(drop=True)
    if working.empty:
        raise ValueError("No analysis-eligible rows were found.")

    working[COMPANY_COLUMN] = working[COMPANY_COLUMN].astype(str).str.strip()
    working[COMMUNITY_COLUMN] = pd.to_numeric(
        working[COMMUNITY_COLUMN], errors="raise"
    ).astype(int)

    for column in COMPANY_METRIC_COLUMNS:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    return working


def identify_communities(dataframe: pd.DataFrame) -> list[int]:
    communities = sorted(
        dataframe[COMMUNITY_COLUMN].dropna().astype(int).unique().tolist()
    )
    if not communities:
        raise ValueError("No communities were found.")
    return communities


# ======================================================================
# PROFILE, PAIR AND RANK CALCULATION
# ======================================================================

def calculate_company_profile(
    company_dataframe: pd.DataFrame,
    company: str,
    communities: list[int],
    replicate_id: int,
    is_observed: bool,
    balanced_sample_size: int,
) -> dict:
    n_songs = len(company_dataframe)
    if n_songs == 0:
        raise ValueError(f"Company '{company}' has no rows.")

    counts = company_dataframe[COMMUNITY_COLUMN].value_counts()
    row = {
        "replicate_id": int(replicate_id),
        "is_observed": bool(is_observed),
        "company": company,
        "n_songs": int(n_songs),
        "balanced_sample_size": int(balanced_sample_size),
    }

    for community in communities:
        row[community_column_name(community)] = float(
            counts.get(community, 0) / n_songs
        )

    for metric in COMPANY_METRIC_COLUMNS:
        values = company_dataframe[metric].dropna().to_numpy(dtype=float)
        row[f"mean_{metric}"] = float(np.mean(values)) if len(values) else np.nan

    fingerprint = np.array(
        [row[community_column_name(c)] for c in communities], dtype=float
    )
    positive = fingerprint[fingerprint > 0]
    entropy = float(-np.sum(positive * np.log(positive)))
    dominant_index = int(np.argmax(fingerprint))

    row["entropy"] = entropy
    row["effective_communities"] = float(np.exp(entropy))
    row["dominant_community"] = int(communities[dominant_index])
    row["dominant_community_share"] = float(fingerprint[dominant_index])
    return row


def calculate_profiles_for_sample(
    sample: pd.DataFrame,
    companies: list[str],
    communities: list[int],
    replicate_id: int,
    is_observed: bool,
    balanced_sample_size: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            calculate_company_profile(
                company_dataframe=sample.loc[sample[COMPANY_COLUMN] == company],
                company=company,
                communities=communities,
                replicate_id=replicate_id,
                is_observed=is_observed,
                balanced_sample_size=balanced_sample_size,
            )
            for company in companies
        ]
    )


def calculate_company_pairs(
    profiles: pd.DataFrame,
    communities: list[int],
) -> pd.DataFrame:
    community_columns = [community_column_name(c) for c in communities]
    replicate_ids = profiles["replicate_id"].unique()
    observed_values = profiles["is_observed"].unique()

    if len(replicate_ids) != 1 or len(observed_values) != 1:
        raise ValueError("Profiles must belong to one consistent replicate.")

    replicate_id = int(replicate_ids[0])
    is_observed = bool(observed_values[0])
    lookup = {
        row["company"]: row[community_columns].to_numpy(dtype=float)
        for _, row in profiles.iterrows()
    }

    rows = []
    for company_a, company_b in combinations(sorted(lookup), 2):
        rows.append(
            {
                "replicate_id": replicate_id,
                "is_observed": is_observed,
                "company_a": company_a,
                "company_b": company_b,
                "company_pair": f"{company_a}|{company_b}",
                "cosine_distance": cosine_distance(
                    lookup[company_a], lookup[company_b]
                ),
            }
        )
    return pd.DataFrame(rows)


def calculate_company_rankings(
    pairs: pd.DataFrame,
    companies: list[str],
) -> pd.DataFrame:
    replicate_ids = pairs["replicate_id"].unique()
    observed_values = pairs["is_observed"].unique()

    if len(replicate_ids) != 1 or len(observed_values) != 1:
        raise ValueError("Pair rows must belong to one consistent replicate.")

    rows = []
    for company in companies:
        distances = pairs.loc[
            (pairs["company_a"] == company) | (pairs["company_b"] == company),
            "cosine_distance",
        ].to_numpy(dtype=float)

        if len(distances) != len(companies) - 1:
            raise ValueError(f"Unexpected pair count for company '{company}'.")

        rows.append(
            {
                "replicate_id": int(replicate_ids[0]),
                "is_observed": bool(observed_values[0]),
                "company": company,
                "mean_pairwise_distance": float(np.mean(distances)),
                "median_pairwise_distance": float(np.median(distances)),
                "maximum_pairwise_distance": float(np.max(distances)),
            }
        )

    rankings = pd.DataFrame(rows)
    rankings["distinctiveness_rank"] = (
        rankings["mean_pairwise_distance"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    return rankings.sort_values(
        ["distinctiveness_rank", "company"]
    ).reset_index(drop=True)


# ======================================================================
# BALANCED SAMPLING
# ======================================================================

def generate_balanced_sample(
    master: pd.DataFrame,
    companies: list[str],
    balanced_sample_size: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    sampled = []

    for company in companies:
        company_data = master.loc[
            master[COMPANY_COLUMN] == company
        ].reset_index(drop=True)

        if len(company_data) < balanced_sample_size:
            raise ValueError(
                f"Company '{company}' contains fewer than "
                f"{balanced_sample_size} rows."
            )

        indices = rng.choice(
            len(company_data), size=balanced_sample_size, replace=False
        )
        sampled.append(company_data.iloc[indices].copy())

    return pd.concat(sampled, ignore_index=True).reset_index(drop=True)


def run_balanced_sampling(
    master: pd.DataFrame,
    companies: list[str],
    communities: list[int],
    balanced_sample_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    profile_frames = []
    pair_frames = []
    ranking_frames = []

    observed_profiles = calculate_profiles_for_sample(
        master, companies, communities, 0, True, balanced_sample_size
    )
    observed_pairs = calculate_company_pairs(observed_profiles, communities)
    observed_rankings = calculate_company_rankings(observed_pairs, companies)

    profile_frames.append(observed_profiles)
    pair_frames.append(observed_pairs)
    ranking_frames.append(observed_rankings)

    progress_interval = max(1, N_REPLICATES // 20)

    for replicate_id in range(1, N_REPLICATES + 1):
        sample = generate_balanced_sample(
            master, companies, balanced_sample_size, rng
        )
        profiles = calculate_profiles_for_sample(
            sample,
            companies,
            communities,
            replicate_id,
            False,
            balanced_sample_size,
        )
        pairs = calculate_company_pairs(profiles, communities)
        rankings = calculate_company_rankings(pairs, companies)

        profile_frames.append(profiles)
        pair_frames.append(pairs)
        ranking_frames.append(rankings)

        if replicate_id % progress_interval == 0 or replicate_id == N_REPLICATES:
            print(f"Balanced progress      : {replicate_id}/{N_REPLICATES}")

    return (
        pd.concat(profile_frames, ignore_index=True),
        pd.concat(pair_frames, ignore_index=True),
        pd.concat(ranking_frames, ignore_index=True),
    )


# ======================================================================
# PROFILE CHANGE SUMMARY
# ======================================================================

def build_profile_changes(
    profiles: pd.DataFrame,
    communities: list[int],
) -> pd.DataFrame:
    community_columns = [community_column_name(c) for c in communities]
    observed = profiles.loc[profiles["is_observed"]].set_index("company")
    replicates = profiles.loc[~profiles["is_observed"]]
    rows = []

    for _, replicate_row in replicates.iterrows():
        company = str(replicate_row["company"])
        observed_row = observed.loc[company]
        observed_vector = observed_row[community_columns].to_numpy(dtype=float)
        replicate_vector = replicate_row[community_columns].to_numpy(dtype=float)
        difference = replicate_vector - observed_vector

        rows.append(
            {
                "replicate_id": int(replicate_row["replicate_id"]),
                "company": company,
                "observed_n_songs": int(observed_row["n_songs"]),
                "balanced_n_songs": int(replicate_row["n_songs"]),
                "is_balancing_reference": bool(
                    int(observed_row["n_songs"])
                    == int(replicate_row["n_songs"])
                ),
                "profile_cosine_distance": cosine_distance(
                    observed_vector, replicate_vector
                ),
                "profile_l1_shift": float(np.sum(np.abs(difference))),
                "profile_l2_shift": float(np.linalg.norm(difference)),
                "maximum_community_share_shift": float(
                    np.max(np.abs(difference))
                ),
                "observed_entropy": float(observed_row["entropy"]),
                "replicate_entropy": float(replicate_row["entropy"]),
                "entropy_change": float(
                    replicate_row["entropy"] - observed_row["entropy"]
                ),
                "observed_effective_communities": float(
                    observed_row["effective_communities"]
                ),
                "replicate_effective_communities": float(
                    replicate_row["effective_communities"]
                ),
                "effective_communities_change": float(
                    replicate_row["effective_communities"]
                    - observed_row["effective_communities"]
                ),
                "observed_dominant_community": int(
                    observed_row["dominant_community"]
                ),
                "replicate_dominant_community": int(
                    replicate_row["dominant_community"]
                ),
                "dominant_community_preserved": bool(
                    replicate_row["dominant_community"]
                    == observed_row["dominant_community"]
                ),
            }
        )

    return pd.DataFrame(rows)


def build_profile_summary(profile_changes: pd.DataFrame) -> pd.DataFrame:
    lower_q, upper_q = confidence_limits()
    rows = []

    for company, data in profile_changes.groupby("company", sort=True):
        rows.append(
            {
                "company": company,
                "observed_n_songs": int(data["observed_n_songs"].iloc[0]),
                "balanced_n_songs": int(data["balanced_n_songs"].iloc[0]),
                "is_balancing_reference": bool(
                    data["is_balancing_reference"].iloc[0]
                ),
                "mean_profile_cosine_distance": float(
                    data["profile_cosine_distance"].mean()
                ),
                "median_profile_cosine_distance": float(
                    data["profile_cosine_distance"].median()
                ),
                "profile_cosine_distance_ci_lower": float(
                    data["profile_cosine_distance"].quantile(lower_q)
                ),
                "profile_cosine_distance_ci_upper": float(
                    data["profile_cosine_distance"].quantile(upper_q)
                ),
                "maximum_profile_cosine_distance": float(
                    data["profile_cosine_distance"].max()
                ),
                "mean_profile_l1_shift": float(data["profile_l1_shift"].mean()),
                "maximum_profile_l1_shift": float(data["profile_l1_shift"].max()),
                "mean_maximum_community_share_shift": float(
                    data["maximum_community_share_shift"].mean()
                ),
                "maximum_community_share_shift": float(
                    data["maximum_community_share_shift"].max()
                ),
                "dominant_community_preservation_probability": float(
                    data["dominant_community_preserved"].mean()
                ),
                "mean_entropy_change": float(data["entropy_change"].mean()),
                "maximum_absolute_entropy_change": float(
                    data["entropy_change"].abs().max()
                ),
            }
        )

    return pd.DataFrame(rows)


# ======================================================================
# PAIR AND RANK SUMMARIES
# ======================================================================

def build_pair_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    observed = pairs.loc[pairs["is_observed"]]
    replicates = pairs.loc[~pairs["is_observed"]]
    rows = []

    for company_pair in sorted(pairs["company_pair"].unique()):
        observed_pair = observed.loc[observed["company_pair"] == company_pair]
        replicate_pair = replicates.loc[
            replicates["company_pair"] == company_pair
        ]

        if len(observed_pair) != 1:
            raise ValueError(f"Expected one observed row for '{company_pair}'.")

        observed_row = observed_pair.iloc[0]
        summary = summarise_numeric_values(
            replicate_pair["cosine_distance"],
            float(observed_row["cosine_distance"]),
        )
        absolute_changes = (
            replicate_pair["cosine_distance"]
            - float(observed_row["cosine_distance"])
        ).abs()

        rows.append(
            {
                "company_a": observed_row["company_a"],
                "company_b": observed_row["company_b"],
                "company_pair": company_pair,
                **summary,
                "mean_absolute_change": float(absolute_changes.mean()),
                "median_absolute_change": float(absolute_changes.median()),
                "maximum_absolute_change": float(absolute_changes.max()),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["observed_value", "company_pair"], ascending=[False, True]
    ).reset_index(drop=True)


def build_ranking_summary(rankings: pd.DataFrame) -> pd.DataFrame:
    observed = rankings.loc[rankings["is_observed"]]
    replicates = rankings.loc[~rankings["is_observed"]]
    companies = sorted(rankings["company"].unique())
    n_companies = len(companies)
    rows = []

    for company in companies:
        observed_company = observed.loc[observed["company"] == company]
        replicate_company = replicates.loc[replicates["company"] == company]

        if len(observed_company) != 1:
            raise ValueError(f"Expected one observed ranking for '{company}'.")

        observed_row = observed_company.iloc[0]
        observed_rank = int(observed_row["distinctiveness_rank"])
        rank_values = replicate_company["distinctiveness_rank"].astype(int)
        distance_summary = summarise_numeric_values(
            replicate_company["mean_pairwise_distance"],
            float(observed_row["mean_pairwise_distance"]),
        )

        row = {
            "company": company,
            "observed_rank": observed_rank,
            "observed_mean_pairwise_distance": float(
                observed_row["mean_pairwise_distance"]
            ),
            "balanced_mean_pairwise_distance": distance_summary["replicate_mean"],
            "distance_ci_lower": distance_summary["ci_lower"],
            "distance_ci_upper": distance_summary["ci_upper"],
            "distance_sampling_bias": distance_summary["sampling_bias"],
            "distance_observed_within_ci": distance_summary["observed_within_ci"],
            "mean_balanced_rank": float(rank_values.mean()),
            "median_balanced_rank": float(rank_values.median()),
            "rank_std": float(rank_values.std(ddof=1)),
            "observed_rank_probability": float(
                (rank_values == observed_rank).mean()
            ),
            "top_rank_probability": float((rank_values == 1).mean()),
            "top_two_probability": float((rank_values <= 2).mean()),
            "bottom_rank_probability": float(
                (rank_values == n_companies).mean()
            ),
            "bottom_two_probability": float(
                (rank_values >= n_companies - 1).mean()
            ),
        }

        for rank in range(1, n_companies + 1):
            row[f"rank_{rank}_probability"] = float(
                (rank_values == rank).mean()
            )

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["observed_rank", "company"]
    ).reset_index(drop=True)


def build_ranking_stability(rankings: pd.DataFrame) -> pd.DataFrame:
    observed = rankings.loc[rankings["is_observed"]].sort_values(
        ["distinctiveness_rank", "company"]
    ).reset_index(drop=True)
    replicates = rankings.loc[~rankings["is_observed"]]

    observed_order = tuple(observed["company"].tolist())
    observed_top_two = frozenset(observed.head(2)["company"].tolist())
    observed_bottom_two = frozenset(observed.tail(2)["company"].tolist())
    observed_rank_lookup = observed.set_index("company")["distinctiveness_rank"]

    exact_matches = 0
    top_two_matches = 0
    bottom_two_matches = 0
    joint_matches = 0
    maximum_rank_displacement = 0
    maximum_changed_company_count = 0

    replicate_ids = sorted(replicates["replicate_id"].unique().tolist())

    for replicate_id in replicate_ids:
        replicate = replicates.loc[
            replicates["replicate_id"] == replicate_id
        ].sort_values(["distinctiveness_rank", "company"]).reset_index(drop=True)

        replicate_order = tuple(replicate["company"].tolist())
        replicate_top_two = frozenset(replicate.head(2)["company"].tolist())
        replicate_bottom_two = frozenset(replicate.tail(2)["company"].tolist())

        exact_match = replicate_order == observed_order
        top_match = replicate_top_two == observed_top_two
        bottom_match = replicate_bottom_two == observed_bottom_two

        exact_matches += int(exact_match)
        top_two_matches += int(top_match)
        bottom_two_matches += int(bottom_match)
        joint_matches += int(top_match and bottom_match)

        replicate_rank_lookup = replicate.set_index("company")[
            "distinctiveness_rank"
        ]
        displacements = (replicate_rank_lookup - observed_rank_lookup).abs()
        maximum_rank_displacement = max(
            maximum_rank_displacement, int(displacements.max())
        )
        maximum_changed_company_count = max(
            maximum_changed_company_count, int((displacements > 0).sum())
        )

    n_replicates = len(replicate_ids)
    if n_replicates == 0:
        raise ValueError("No balanced-sampling replicates were found.")

    exact_probability = float(exact_matches / n_replicates)
    random_probability = float(1.0 / factorial(len(observed_order)))

    return pd.DataFrame(
        [
            {
                "n_replicates": n_replicates,
                "n_companies": len(observed_order),
                "observed_order": "|".join(observed_order),
                "observed_top_two": "|".join(sorted(observed_top_two)),
                "observed_bottom_two": "|".join(sorted(observed_bottom_two)),
                "exact_order_probability": exact_probability,
                "random_exact_order_probability": random_probability,
                "exact_order_stability_ratio": exact_probability / random_probability,
                "top_two_set_probability": top_two_matches / n_replicates,
                "bottom_two_set_probability": bottom_two_matches / n_replicates,
                "joint_group_probability": joint_matches / n_replicates,
                "maximum_rank_displacement": maximum_rank_displacement,
                "maximum_changed_company_count": maximum_changed_company_count,
            }
        ]
    )


# ======================================================================
# ROBUSTNESS CLASSIFICATION
# ======================================================================

def classify_rank_robustness(probability: float) -> str:
    if probability >= 0.90:
        return "highly_robust"
    if probability >= 0.75:
        return "robust"
    if probability >= 0.50:
        return "moderately_robust"
    return "rank_sensitive"


def classify_profile_robustness(mean_shift: float, upper_shift: float) -> str:
    if mean_shift <= 0.010 and upper_shift <= 0.025:
        return "highly_robust"
    if mean_shift <= 0.025 and upper_shift <= 0.050:
        return "robust"
    if mean_shift <= 0.050:
        return "moderately_robust"
    return "profile_sensitive"


def classify_pair_robustness(
    mean_change: float,
    maximum_change: float,
    observed_within_ci: bool,
) -> str:
    if not observed_within_ci:
        return "pair_sensitive"
    if mean_change <= 0.010 and maximum_change <= 0.050:
        return "highly_robust"
    if mean_change <= 0.025 and maximum_change <= 0.100:
        return "robust"
    if mean_change <= 0.050:
        return "moderately_robust"
    return "pair_sensitive"


def classify_global_rank_robustness(
    exact_probability: float,
    stability_ratio: float,
) -> str:
    if exact_probability >= 0.80 and stability_ratio >= 10.0:
        return "highly_robust"
    if exact_probability >= 0.60 and stability_ratio >= 8.0:
        return "robust"
    if stability_ratio >= 4.0:
        return "moderately_robust"
    return "rank_sensitive"


def classify_global_profile_structure(
    profile_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
) -> str:
    max_mean_profile_shift = float(
        profile_summary["mean_profile_cosine_distance"].max()
    )
    all_pairs_inside_ci = bool(pair_summary["observed_within_ci"].all())

    if max_mean_profile_shift <= 0.010 and all_pairs_inside_ci:
        return "highly_robust"
    if max_mean_profile_shift <= 0.025 and all_pairs_inside_ci:
        return "robust"
    if max_mean_profile_shift <= 0.050:
        return "moderately_robust"
    return "profile_sensitive"


# ======================================================================
# PUBLICATION SUMMARY
# ======================================================================

def build_publication_summary(
    profile_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    ranking_summary: pd.DataFrame,
    ranking_stability: pd.DataFrame,
) -> pd.DataFrame:
    if len(ranking_stability) != 1:
        raise ValueError("Expected one global ranking-stability row.")

    global_row = ranking_stability.iloc[0]
    rows = [
        {
            "analysis": "Global balanced-sampling robustness",
            "entity": "All companies",
            "finding_tested": (
                "Continuous semantic structure and complete distinctiveness ordering"
            ),
            "observed_value": np.nan,
            "balanced_mean": np.nan,
            "primary_probability": float(global_row["exact_order_probability"]),
            "secondary_test": "Observed top-two and bottom-two grouping",
            "secondary_probability": float(global_row["joint_group_probability"]),
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "mean_absolute_change": np.nan,
            "maximum_absolute_change": float(
                global_row["maximum_rank_displacement"]
            ),
            "stability_ratio": float(
                global_row["exact_order_stability_ratio"]
            ),
            "is_balancing_reference": np.nan,
            "profile_conclusion": classify_global_profile_structure(
                profile_summary, pair_summary
            ),
            "ranking_conclusion": classify_global_rank_robustness(
                float(global_row["exact_order_probability"]),
                float(global_row["exact_order_stability_ratio"]),
            ),
            "pairwise_conclusion": np.nan,
            "interpretation": (
                "Catalogue balancing affects exact ordinal positions more strongly "
                "than the underlying continuous company-profile structure."
            ),
        }
    ]

    profile_lookup = profile_summary.set_index("company")

    for _, ranking_row in ranking_summary.iterrows():
        company = str(ranking_row["company"])
        profile_row = profile_lookup.loc[company]
        is_reference = bool(profile_row["is_balancing_reference"])

        rows.append(
            {
                "analysis": "Company balanced-sampling robustness",
                "entity": company,
                "finding_tested": "Company semantic profile and distinctiveness rank",
                "observed_value": float(ranking_row["observed_rank"]),
                "balanced_mean": float(
                    ranking_row["balanced_mean_pairwise_distance"]
                ),
                "primary_probability": float(
                    ranking_row["observed_rank_probability"]
                ),
                "secondary_test": "Dominant community preserved",
                "secondary_probability": float(
                    profile_row["dominant_community_preservation_probability"]
                ),
                "ci_lower": float(ranking_row["distance_ci_lower"]),
                "ci_upper": float(ranking_row["distance_ci_upper"]),
                "mean_absolute_change": float(
                    profile_row["mean_profile_cosine_distance"]
                ),
                "maximum_absolute_change": float(
                    profile_row["maximum_profile_cosine_distance"]
                ),
                "stability_ratio": np.nan,
                "is_balancing_reference": is_reference,
                "profile_conclusion": classify_profile_robustness(
                    float(profile_row["mean_profile_cosine_distance"]),
                    float(profile_row["profile_cosine_distance_ci_upper"]),
                ),
                "ranking_conclusion": classify_rank_robustness(
                    float(ranking_row["observed_rank_probability"])
                ),
                "pairwise_conclusion": np.nan,
                "interpretation": (
                    "This company defines the balancing threshold, so its full "
                    "profile is retained in every replicate; rank movement is "
                    "caused by variation in the other companies."
                    if is_reference
                    else
                    "Profile stability and rank stability are reported separately."
                ),
            }
        )

    for _, pair_row in pair_summary.iterrows():
        rows.append(
            {
                "analysis": "Pairwise balanced-sampling robustness",
                "entity": str(pair_row["company_pair"]),
                "finding_tested": "Observed company-profile cosine distance",
                "observed_value": float(pair_row["observed_value"]),
                "balanced_mean": float(pair_row["replicate_mean"]),
                "primary_probability": np.nan,
                "secondary_test": "Observed estimate inside balanced 95% interval",
                "secondary_probability": float(
                    bool(pair_row["observed_within_ci"])
                ),
                "ci_lower": float(pair_row["ci_lower"]),
                "ci_upper": float(pair_row["ci_upper"]),
                "mean_absolute_change": float(pair_row["mean_absolute_change"]),
                "maximum_absolute_change": float(
                    pair_row["maximum_absolute_change"]
                ),
                "stability_ratio": np.nan,
                "is_balancing_reference": np.nan,
                "profile_conclusion": np.nan,
                "ranking_conclusion": np.nan,
                "pairwise_conclusion": classify_pair_robustness(
                    float(pair_row["mean_absolute_change"]),
                    float(pair_row["maximum_absolute_change"]),
                    bool(pair_row["observed_within_ci"]),
                ),
                "interpretation": (
                    "Pairwise robustness uses distance change and interval coverage."
                ),
            }
        )

    return pd.DataFrame(rows)


# ======================================================================
# CONSOLE OUTPUT
# ======================================================================

def print_results(
    master: pd.DataFrame,
    company_counts: pd.Series,
    balanced_sample_size: int,
    communities: list[int],
    profile_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    ranking_summary: pd.DataFrame,
    ranking_stability: pd.DataFrame,
    publication_summary: pd.DataFrame,
) -> None:
    print_heading("WP6 : BALANCED COMPANY SAMPLING")
    print(f"Master dataset        : {MASTER_PATH}")
    print(f"Eligible master rows  : {len(master)}")
    print(f"Companies analysed    : {len(company_counts)}")
    print(f"Communities analysed  : {len(communities)}")
    print(f"Balanced sample size  : {balanced_sample_size}")
    print(f"Songs per replicate   : {balanced_sample_size * len(company_counts)}")
    print(f"Replicates            : {N_REPLICATES}")
    print(f"Random seed           : {RANDOM_SEED}")

    print_heading("ORIGINAL COMPANY SAMPLE SIZES")
    print(company_counts.rename("n_songs").reset_index().to_string(index=False))

    global_row = ranking_stability.iloc[0]
    print_heading("GLOBAL RANKING STABILITY")
    print(f"Observed order           : {global_row['observed_order']}")
    print(f"Exact-order probability  : {global_row['exact_order_probability']:.4f}")
    print(f"Random-order probability : {global_row['random_exact_order_probability']:.4f}")
    print(f"Order stability ratio    : {global_row['exact_order_stability_ratio']:.4f}")
    print(f"Top-two-set probability  : {global_row['top_two_set_probability']:.4f}")
    print(f"Bottom-two probability   : {global_row['bottom_two_set_probability']:.4f}")
    print(f"Joint grouping probability: {global_row['joint_group_probability']:.4f}")
    print(f"Maximum rank displacement: {global_row['maximum_rank_displacement']}")

    print_heading("COMPANY RANKING SUMMARY")
    print(
        ranking_summary[
            [
                "company",
                "observed_rank",
                "observed_mean_pairwise_distance",
                "balanced_mean_pairwise_distance",
                "distance_ci_lower",
                "distance_ci_upper",
                "observed_rank_probability",
                "top_rank_probability",
                "top_two_probability",
                "bottom_two_probability",
            ]
        ].to_string(index=False)
    )

    print_heading("COMPANY PROFILE SHIFT SUMMARY")
    print(
        profile_summary[
            [
                "company",
                "observed_n_songs",
                "balanced_n_songs",
                "is_balancing_reference",
                "mean_profile_cosine_distance",
                "profile_cosine_distance_ci_lower",
                "profile_cosine_distance_ci_upper",
                "maximum_profile_cosine_distance",
                "dominant_community_preservation_probability",
            ]
        ].to_string(index=False)
    )

    print_heading("PAIRWISE DISTANCE SUMMARY")
    print(
        pair_summary[
            [
                "company_pair",
                "observed_value",
                "replicate_mean",
                "replicate_std",
                "ci_lower",
                "ci_upper",
                "mean_absolute_change",
                "maximum_absolute_change",
                "observed_within_ci",
            ]
        ].to_string(index=False)
    )

    print_heading("PUBLICATION SUMMARY")
    print(publication_summary.to_string(index=False))


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    print_heading("WP6 : BALANCED COMPANY SAMPLING")

    if not MASTER_PATH.exists():
        raise FileNotFoundError(
            "The authoritative master dataset does not exist: "
            f"{MASTER_PATH}"
        )

    master_raw = pd.read_parquet(MASTER_PATH)
    validate_master_dataset(master_raw)
    master = prepare_master_dataset(master_raw)

    companies = sorted(master[COMPANY_COLUMN].unique().tolist())
    communities = identify_communities(master)
    company_counts = master.groupby(COMPANY_COLUMN).size().sort_index()

    if len(companies) < 2:
        raise ValueError("At least two companies are required.")

    balanced_sample_size = int(company_counts.min())
    if balanced_sample_size < 2:
        raise ValueError("The smallest company contains fewer than two songs.")

    balancing_reference = company_counts.loc[
        company_counts == balanced_sample_size
    ].index.tolist()

    print(f"Master dataset        : {MASTER_PATH}")
    print(f"Master rows loaded    : {len(master_raw)}")
    print(f"Eligible master rows  : {len(master)}")
    print(f"Companies analysed    : {len(companies)}")
    print(f"Communities analysed  : {len(communities)}")
    print(f"Balanced sample size  : {balanced_sample_size}")
    print(f"Balancing reference   : {'|'.join(balancing_reference)}")
    print(f"Replicates            : {N_REPLICATES}")

    profiles, pairs, rankings = run_balanced_sampling(
        master, companies, communities, balanced_sample_size
    )
    profile_changes = build_profile_changes(profiles, communities)
    profile_summary = build_profile_summary(profile_changes)
    pair_summary = build_pair_summary(pairs)
    ranking_summary = build_ranking_summary(rankings)
    ranking_stability = build_ranking_stability(rankings)
    publication_summary = build_publication_summary(
        profile_summary,
        pair_summary,
        ranking_summary,
        ranking_stability,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(BALANCED_PROFILES_PATH, index=False)
    pairs.to_csv(BALANCED_PAIRS_PATH, index=False)
    rankings.to_csv(BALANCED_RANKINGS_PATH, index=False)
    profile_changes.to_csv(PROFILE_CHANGES_PATH, index=False)
    profile_summary.to_csv(PROFILE_SUMMARY_PATH, index=False)
    pair_summary.to_csv(PAIR_SUMMARY_PATH, index=False)
    ranking_summary.to_csv(RANKING_SUMMARY_PATH, index=False)
    ranking_stability.to_csv(RANKING_STABILITY_PATH, index=False)
    publication_summary.to_csv(PUBLICATION_SUMMARY_PATH, index=False)

    print_results(
        master,
        company_counts,
        balanced_sample_size,
        communities,
        profile_summary,
        pair_summary,
        ranking_summary,
        ranking_stability,
        publication_summary,
    )

    print_heading("FILES WRITTEN")
    for output_path in [
        BALANCED_PROFILES_PATH,
        BALANCED_PAIRS_PATH,
        BALANCED_RANKINGS_PATH,
        PROFILE_CHANGES_PATH,
        PROFILE_SUMMARY_PATH,
        PAIR_SUMMARY_PATH,
        RANKING_SUMMARY_PATH,
        RANKING_STABILITY_PATH,
        PUBLICATION_SUMMARY_PATH,
    ]:
        print(output_path)
    print("=" * 70)


if __name__ == "__main__":
    main()
