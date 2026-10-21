"""
02_company_statistics.py

Publication-quality descriptive statistics for the
Big Four semantic analysis.

Outputs
-------
1. Song-level descriptive statistics
2. Artist-level descriptive statistics

Author: Oktay Karakus
"""

from pathlib import Path

import pandas as pd
import sys


###############################################################################
# Paths
###############################################################################

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))
print(f"ROOT: {ROOT}")
from utils.statistics import descriptive_statistics


INPUT = (
    ROOT
    / "outputs"
    / "tables"
    / "company_master_analysis_v1.parquet"
)

OUTPUT = (
    ROOT
    / "outputs"
    / "tables"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

###############################################################################
# Load
###############################################################################

print("=" * 80)
print("BIG FOUR COMPANY STATISTICS")
print("=" * 80)

df = pd.read_parquet(INPUT)

print(f"Songs : {len(df):,}")
print(f"Artists : {df.artist.nunique():,}")
print()

###############################################################################
# Variables
###############################################################################

VARIABLES = [
    "participation_coefficient",
    "community_entropy",
    "centroid_novelty",
    "local_novelty",
]

###############################################################################
# SONG LEVEL
###############################################################################

print("=" * 80)
print("SONG-LEVEL DESCRIPTIVE STATISTICS")
print("=" * 80)

song_tables = []

for variable in VARIABLES:

    stats = descriptive_statistics(
        df=df,
        value=variable,
        group="company",
    )

    stats.insert(0, "variable", variable)

    song_tables.append(stats)

song_results = pd.concat(song_tables, ignore_index=True)

song_results.to_csv(
    OUTPUT / "song_level_statistics.csv",
    index=False,
)

print(song_results.head())
print()

###############################################################################
# ARTIST LEVEL
###############################################################################

print("=" * 80)
print("ARTIST-LEVEL DESCRIPTIVE STATISTICS")
print("=" * 80)


artist_df = (
    df.groupby(
        ["company", "artist"],
        as_index=False,
        observed=True,
    )[VARIABLES]
)

artist_tables = []

for variable in VARIABLES:

    stats = descriptive_statistics(
        df=artist_df,
        value=variable,
        group="company",
    )

    stats.insert(0, "variable", variable)

    artist_tables.append(stats)

artist_results = pd.concat(
    artist_tables,
    ignore_index=True,
)

artist_results.to_csv(
    OUTPUT / "artist_level_statistics.csv",
    index=False,
)

print(artist_results.head())

###############################################################################
# Summary
###############################################################################

print()
print("=" * 80)
print("SAVED")
print("=" * 80)

print(OUTPUT / "song_level_statistics.csv")
print(OUTPUT / "artist_level_statistics.csv")