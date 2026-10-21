#!/usr/bin/env python3
"""
PLOS-C1: derive the text-free candidate dataset from the private analysis table.

Input (private, NOT for release; contains full lyrics):
    outputs/tables/company_analysis_dataset.parquet   (1,289 x 36)

Output (release candidate):
    data/company_text_free_v1.parquet
    data/company_text_free_v1.csv
    data/data_dictionary.csv

No statistics are recomputed. Rows are copied one-to-one; columns are selected
and two identifiers are replaced by deterministic pseudonyms:
    record_id  = "S" + 4-digit rank of song_uid in ascending sort order
    artist_id  = "A" + 3-digit rank of artist_normalised in ascending sort order
The crosswalk can be regenerated only from the private input, so no crosswalk
file is written.

Usage:
    python derive_text_free_dataset.py <input_parquet> <output_data_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

KEEP = [
    # (output column, source column, description)
    ("record_id", None, "Pseudonymous song record ID (replaces song_uid; see derivation rule in README)"),
    ("artist_id", None, "Pseudonymous artist ID (replaces artist_normalised; grouping unit for artist-level tests and leave-one-artist-out)"),
    ("company", "company", "Big Four company label (static portfolio; not verified at release date)"),
    ("release_year", "release_year", "Release year (analysis year; 'year' in company_master_analysis_v1.parquet)"),
    ("analysis_eligible", "analysis_eligible", "Eligibility flag used as a filter by the analysis scripts (True for all rows)"),
    ("community", "community", "Semantic community label (15 communities)"),
    ("degree", "degree", "Similarity-graph degree"),
    ("weighted_degree", "weighted_degree", "Similarity-graph weighted degree"),
    ("external_strength", "external_strength", "Edge weight to songs outside own community"),
    ("external_strength_fraction", "external_strength_fraction", "external_strength / weighted_degree"),
    ("participation_coefficient", "participation_coefficient", "Participation coefficient across communities"),
    ("community_entropy", "community_entropy", "Entropy of neighbour-community distribution"),
    ("external_communities_reached", "external_communities_reached", "Number of other communities reached by edges"),
    ("local_novelty", "local_novelty", "1 - mean cosine similarity to graph neighbours"),
    ("max_centroid_cosine", "max_centroid_cosine", "Cosine similarity to nearest community centroid"),
    ("centroid_margin", "centroid_margin", "Gap between nearest and second-nearest centroid cosine"),
    ("centroid_novelty", "centroid_novelty", "1 - max_centroid_cosine"),
    ("bridge_affinity_score", "bridge_affinity_score", "Bridge affinity score"),
]

EXCLUDED = {
    "lyrics": "Full copyrighted lyric text",
    "lines": "Full copyrighted lyric text (line arrays)",
    "title": "Song title; not used by any reported statistic",
    "artist": "Original-script duplicate of artist_normalised",
    "artist_normalised": "Replaced by pseudonymous artist_id (author decision pending on restoring names)",
    "song_uid": "Encodes Melon chart file/month/rank; replaced by pseudonymous record_id",
    "source_file": "Melon chart-file path; third-party provenance, redistribution terms unknown",
    "date": "Full release date; release_year suffices for all analyses",
    "release_period": "Not read by any analysis or plotting script; derivable from release_year",
    "mapping_status": "Constant ('matched_student_lookup'); documented in README",
    "mapping_basis": "Constant ('static_student_lookup'); documented in README",
    "company_at_release_verified": "Constant (False); documented in README",
    "mapping_review_required": "Constant (False); documented in README",
    "mapping_review_reason": "Empty for all rows",
    "internal_strength": "Not read by analysis scripts; equals weighted_degree - external_strength",
    "communities_reached": "Not read by analysis scripts; equals external_communities_reached + 1",
    "mean_neighbour_cosine": "Not read by analysis scripts; equals 1 - local_novelty",
    "second_centroid_cosine": "Not read by analysis scripts; equals max_centroid_cosine - centroid_margin",
    "n_lines": "Lyric-derived size descriptor; not used by any reported statistic",
    "n_chars": "Lyric-derived size descriptor; not used by any reported statistic",
}


def main(src: Path, out_dir: Path) -> None:
    d = pd.read_parquet(src)
    assert d.shape == (1289, 36), d.shape
    assert set(d.columns) == {k for k, s, _ in KEEP if s} | set(EXCLUDED), "column set changed"
    assert d["song_uid"].is_unique

    # Redundancy checks for dropped numeric columns (max abs deviation).
    checks = {
        "internal_strength": (d["weighted_degree"] - d["external_strength"] - d["internal_strength"]).abs().max(),
        "communities_reached": (d["external_communities_reached"] + 1 - d["communities_reached"]).abs().max(),
        "mean_neighbour_cosine": (1 - d["local_novelty"] - d["mean_neighbour_cosine"]).abs().max(),
        "second_centroid_cosine": (d["max_centroid_cosine"] - d["centroid_margin"] - d["second_centroid_cosine"]).abs().max(),
    }
    for k, v in checks.items():
        print(f"redundancy {k}: max abs deviation = {v:.3e}")
        assert v < 1e-6, k

    uid_rank = {u: i + 1 for i, u in enumerate(sorted(d["song_uid"]))}
    art_rank = {a: i + 1 for i, a in enumerate(sorted(d["artist_normalised"].unique()))}
    out = pd.DataFrame({
        "record_id": d["song_uid"].map(lambda u: f"S{uid_rank[u]:04d}"),
        "artist_id": d["artist_normalised"].map(lambda a: f"A{art_rank[a]:03d}"),
    })
    for name, source, _ in KEEP:
        if source:
            out[name] = d[source].values
    out["release_year"] = out["release_year"].astype("int64")
    out = out.sort_values("record_id").reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_dir / "company_text_free_v1.parquet", index=False)
    out.to_csv(out_dir / "company_text_free_v1.csv", index=False, float_format="%.17g")
    pd.DataFrame(
        [{"column": n, "dtype": str(out[n].dtype), "description": desc} for n, _, desc in KEEP]
    ).to_csv(out_dir / "data_dictionary.csv", index=False)
    print("written", out.shape)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
