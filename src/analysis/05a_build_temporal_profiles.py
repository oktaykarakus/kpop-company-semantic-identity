#!/usr/bin/env python3
"""
=========================================================
WP5 - Temporal Dynamics
05a_build_temporal_profiles.py

Creates annual and rolling 5-year semantic profiles
for each entertainment company.

Input:
    outputs/tables/company_master_analysis_v1.parquet

Outputs:
    outputs/temporal/annual/community_profiles.parquet
    outputs/temporal/rolling5/community_profiles.parquet

Author:
    Oktay Karakus
=========================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy


# --------------------------------------------------------
# Paths
# --------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "outputs" / "tables" / "company_master_analysis_v1.parquet"

OUTPUT_ROOT = ROOT / "outputs" / "temporal"

ANNUAL_DIR = OUTPUT_ROOT / "annual"
ROLLING_DIR = OUTPUT_ROOT / "rolling5"

ANNUAL_DIR.mkdir(parents=True, exist_ok=True)
ROLLING_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------
# Load data
# --------------------------------------------------------

print("=" * 70)
print("WP5 : TEMPORAL PROFILE GENERATOR")
print("=" * 70)

df = pd.read_parquet(INPUT_FILE)

df = df[df["analysis_eligible"]].copy()

print(f"Songs      : {len(df):,}")
print(f"Artists    : {df['artist_normalised'].nunique():,}")
print(f"Companies  : {df['company'].nunique()}")

years = sorted(df["year"].dropna().astype(int).unique())

print(f"Years      : {years[0]} - {years[-1]}")

companies = sorted(df["company"].astype(str).unique())
communities = sorted(df["community"].unique())


# --------------------------------------------------------
# Window generators
# --------------------------------------------------------

def build_annual_windows(years):

    windows = []

    for y in years:

        windows.append(
            {
                "window": str(y),
                "start": y,
                "end": y,
            }
        )

    return windows


def build_rolling_windows(years, width=5):

    windows = []

    first = years[0]
    last = years[-1]

    for start in range(first, last - width + 2):

        windows.append(
            {
                "window": f"{start}-{start+width-1}",
                "start": start,
                "end": start + width - 1,
            }
        )

    return windows


annual_windows = build_annual_windows(years)
rolling_windows = build_rolling_windows(years)

print(f"Annual windows  : {len(annual_windows)}")
print(f"Rolling windows : {len(rolling_windows)}")


# --------------------------------------------------------
# Build company profile
# --------------------------------------------------------

def build_profile(company_df):

    profile = {}

    profile["n_songs"] = len(company_df)

    profile["n_artists"] = company_df["artist_normalised"].nunique()

    profile["mean_degree"] = company_df["degree"].mean()

    profile["mean_weighted_degree"] = company_df["weighted_degree"].mean()

    profile["mean_bridge_affinity"] = company_df["bridge_affinity_score"].mean()

    profile["mean_local_novelty"] = company_df["local_novelty"].mean()

    profile["mean_centroid_novelty"] = company_df["centroid_novelty"].mean()

    counts = (
        company_df["community"]
        .value_counts()
        .reindex(communities, fill_value=0)
    )

    probs = counts / counts.sum()

    H = entropy(probs)

    profile["entropy"] = H
    profile["effective_communities"] = np.exp(H)

    dominant = probs.idxmax()

    profile["dominant_community"] = dominant

    for c in communities:

        profile[f"community_{c}"] = probs.loc[c]

    return profile


# --------------------------------------------------------
# Process windows
# --------------------------------------------------------

def process_windows(windows):

    rows = []

    for window in windows:

        subset = df[
            (df["year"] >= window["start"])
            &
            (df["year"] <= window["end"])
        ]

        for company in companies:

            company_df = subset[
                subset["company"].astype(str) == company
            ]

            if len(company_df) == 0:
                continue

            profile = build_profile(company_df)

            profile["company"] = company
            profile["window"] = window["window"]
            profile["start_year"] = window["start"]
            profile["end_year"] = window["end"]

            rows.append(profile)

    return pd.DataFrame(rows)


print()
print("Building annual profiles...")

annual_profiles = process_windows(annual_windows)

print("Building rolling profiles...")

rolling_profiles = process_windows(rolling_windows)


# --------------------------------------------------------
# Save
# --------------------------------------------------------

annual_file = ANNUAL_DIR / "community_profiles.parquet"

rolling_file = ROLLING_DIR / "community_profiles.parquet"

annual_profiles.to_parquet(
    annual_file,
    index=False,
)

rolling_profiles.to_parquet(
    rolling_file,
    index=False,
)

print()
print("=" * 70)

print("Finished.")

print()

print("Annual profiles :", len(annual_profiles))
print("Rolling profiles:", len(rolling_profiles))

print()

print(annual_file)
print(rolling_file)

print("=" * 70)