"""
09b_evidence_traceability_network.py

WP9b — Evidence Traceability Network
Project: Semantic Fingerprints of K-pop Entertainment Companies

Scientific question
-------------------
Can every frozen WP8 company-identity claim be traced transparently to the
empirical work packages, evidence records, and source files that support or
qualify it?

Principles
----------
- WP1-WP8 and WP9a are treated as frozen.
- No new embeddings, semantic graphs, clustering, prediction models, latent
  variables, inferential tests, or composite scores are created.
- The "network" is a deterministic provenance structure, not a newly learned
  analytical graph.
- WP9a evidence records are reused without changing their scientific status.
- Raw source-level traceability is retained.
- Duplicate source records are flagged but not deleted.
- Orphaned nodes and broken links are reported explicitly.

Canonical inputs
----------------
outputs/framework_validation/09a4_cross_workpackage_consistency/
    cross_wp_evidence_ledger.csv
    cross_wp_consistency_summary.csv
    cross_wp_input_manifest.csv
    cross_wp_run_summary.csv

Exact WP8 filenames resolved uniquely within outputs/:
    integrated_company_identity.csv
    company_identity_summary.csv
    identity_evidence_traceability.csv
    cross_analysis_consistency.csv

Outputs
-------
outputs/framework_validation/09b_evidence_traceability_network/
    traceability_nodes.csv
    traceability_edges.csv
    claim_evidence_traceability.csv
    claim_traceability_summary.csv
    workpackage_claim_coverage.csv
    source_file_traceability.csv
    traceability_orphans.csv
    traceability_input_manifest.csv
    traceability_run_summary.csv

Run
---
python 09b_evidence_traceability_network.py
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
import re
import sys
from typing import Iterable

import numpy as np
import pandas as pd


# =============================================================================
# Project paths
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

WP9A_DIR = (
    ROOT
    / "outputs"
    / "framework_validation"
    / "09a4_cross_workpackage_consistency"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "framework_validation"
    / "09b_evidence_traceability_network"
)

COMPANIES = ("HYBE", "JYP", "SM", "YG")
EMPIRICAL_WORK_PACKAGES = ("WP3", "WP4", "WP5", "WP6", "WP7")

WP9A_INPUTS = {
    "WP9a_evidence_ledger": WP9A_DIR / "cross_wp_evidence_ledger.csv",
    "WP9a_consistency_summary": WP9A_DIR / "cross_wp_consistency_summary.csv",
    "WP9a_input_manifest": WP9A_DIR / "cross_wp_input_manifest.csv",
    "WP9a_run_summary": WP9A_DIR / "cross_wp_run_summary.csv",
}

WP8_FILENAMES = {
    "WP8_integrated_identity": "integrated_company_identity.csv",
    "WP8_company_summary": "company_identity_summary.csv",
    "WP8_traceability": "identity_evidence_traceability.csv",
    "WP8_cross_consistency": "cross_analysis_consistency.csv",
}


# =============================================================================
# General helpers
# =============================================================================


def normalise_text(value: object) -> str:
    """Return a stable, whitespace-normalised string."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def safe_numeric(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def stable_id(prefix: str, *parts: object, length: int = 16) -> str:
    payload = "||".join(normalise_text(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def read_csv_required(path: Path, required_columns: Iterable[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    data = pd.read_csv(path)
    missing = sorted(set(required_columns) - set(data.columns))
    if missing:
        raise ValueError(
            f"{path.relative_to(ROOT)} is missing required columns: {missing}"
        )
    return data


def resolve_exact_filename(filename: str) -> Path:
    """Resolve exactly one matching filename below outputs/."""
    output_root = ROOT / "outputs"
    matches = sorted(path for path in output_root.rglob(filename) if path.is_file())
    if not matches:
        raise FileNotFoundError(
            f"Could not resolve exact filename below outputs/: {filename}"
        )
    if len(matches) > 1:
        relative = [str(path.relative_to(ROOT)) for path in matches]
        raise RuntimeError(
            f"Exact filename is ambiguous: {filename}. Matches: {relative}"
        )
    return matches[0]


def pipe_join(values: Iterable[object]) -> str:
    cleaned = sorted(
        {
            normalise_text(value)
            for value in values
            if normalise_text(value)
        }
    )
    return "|".join(cleaned)


def json_dumps_compact(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# =============================================================================
# Input loading and validation
# =============================================================================


def resolve_inputs() -> dict[str, Path]:
    paths = dict(WP9A_INPUTS)
    for key, filename in WP8_FILENAMES.items():
        paths[key] = resolve_exact_filename(filename)
    return paths


def build_input_manifest(paths: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, path in paths.items():
        rows.append(
            {
                "input_key": key,
                "work_package": key.split("_", 1)[0],
                "relative_path": str(path.relative_to(ROOT)),
                "exists": path.exists(),
                "resolution_method": (
                    "verified_exact_path"
                    if key in WP9A_INPUTS
                    else "unique_exact_filename"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["work_package", "input_key"])


def load_inputs(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    ledger = read_csv_required(
        paths["WP9a_evidence_ledger"],
        [
            "evidence_id",
            "company",
            "work_package",
            "analysis_component",
            "claim_category",
            "originating_claim",
            "evidence_statement",
            "metric",
            "value",
            "comparison",
            "direction",
            "statistical_status",
            "evidence_status",
            "evidence_relevance",
            "rationale",
            "source_file",
            "source_row",
        ],
    )

    consistency = read_csv_required(
        paths["WP9a_consistency_summary"],
        [
            "claim_id",
            "company",
            "claim_category",
            "major_conclusion",
            "overall_consistency",
        ],
    )

    wp8_traceability = pd.read_csv(paths["WP8_traceability"])
    wp8_identity = pd.read_csv(paths["WP8_integrated_identity"])
    wp8_company = pd.read_csv(paths["WP8_company_summary"])
    wp8_consistency = pd.read_csv(paths["WP8_cross_consistency"])

    return {
        "ledger": ledger,
        "consistency": consistency,
        "wp8_traceability": wp8_traceability,
        "wp8_identity": wp8_identity,
        "wp8_company": wp8_company,
        "wp8_consistency": wp8_consistency,
    }


# =============================================================================
# Claim and evidence preparation
# =============================================================================


def build_claim_table(
    ledger: pd.DataFrame,
    consistency: pd.DataFrame,
) -> pd.DataFrame:
    """Create the canonical WP8 claim table used by the provenance network."""
    ledger_claims = (
        ledger[["company", "originating_claim", "claim_category"]]
        .drop_duplicates()
        .rename(columns={"originating_claim": "claim_text"})
    )
    ledger_claims = ledger_claims[ledger_claims["claim_text"].map(normalise_text) != ""]

    summary_claims = consistency[
        [
            "claim_id",
            "company",
            "claim_category",
            "major_conclusion",
            "overall_consistency",
        ]
    ].rename(columns={"major_conclusion": "claim_text"})

    claims = ledger_claims.merge(
        summary_claims,
        on=["company", "claim_category", "claim_text"],
        how="left",
        validate="many_to_one",
    )

    # Four unmapped WP8 narrative fragments are intentionally absent from the
    # WP9a headline summary. They still receive stable claim IDs here.
    missing_id = claims["claim_id"].isna()
    claims.loc[missing_id, "claim_id"] = claims.loc[missing_id].apply(
        lambda row: stable_id(
            "CLAIM",
            row["company"],
            row["claim_category"],
            row["claim_text"],
        ),
        axis=1,
    )
    claims.loc[missing_id, "overall_consistency"] = "not_evaluated_unmapped"

    claims["is_mapped_claim"] = claims["claim_category"].ne("other")
    claims["claim_node_id"] = claims["claim_id"].map(lambda value: f"CLAIM::{value}")

    if claims["claim_node_id"].duplicated().any():
        duplicates = claims.loc[
            claims["claim_node_id"].duplicated(keep=False),
            ["claim_node_id", "company", "claim_text"],
        ]
        raise RuntimeError(
            "Claim node IDs are not unique:\n" + duplicates.to_string(index=False)
        )

    return claims.sort_values(["company", "claim_category", "claim_text"]).reset_index(drop=True)


def enrich_ledger_with_claim_ids(
    ledger: pd.DataFrame,
    claims: pd.DataFrame,
) -> pd.DataFrame:
    enriched = ledger.merge(
        claims[
            [
                "claim_id",
                "claim_node_id",
                "company",
                "claim_category",
                "claim_text",
                "is_mapped_claim",
                "overall_consistency",
            ]
        ],
        left_on=["company", "claim_category", "originating_claim"],
        right_on=["company", "claim_category", "claim_text"],
        how="left",
        validate="many_to_one",
    )

    if enriched["claim_node_id"].isna().any():
        bad = enriched.loc[
            enriched["claim_node_id"].isna(),
            ["evidence_id", "company", "claim_category", "originating_claim"],
        ]
        raise RuntimeError(
            "Evidence records could not be linked to claims:\n"
            + bad.head(20).to_string(index=False)
        )

    enriched["evidence_node_id"] = enriched["evidence_id"].map(
        lambda value: f"EVIDENCE::{value}"
    )
    enriched["work_package_node_id"] = enriched["work_package"].map(
        lambda value: f"WORK_PACKAGE::{value}"
    )
    enriched["source_node_id"] = enriched["source_file"].map(
        lambda value: stable_id("SOURCE", value)
    )
    enriched["company_node_id"] = enriched["company"].map(
        lambda value: f"COMPANY::{value}"
    )

    # Reproduce the frozen WP9a4 semantic deduplication rule exactly.
    # The two WP3 company-summary tables can repeat the same frozen statistic,
    # so their analysis-component label is collapsed for semantic identity only.
    enriched["semantic_analysis_component"] = enriched["analysis_component"].astype(str)
    wp3_company_summary = (
        enriched["work_package"].eq("WP3")
        & enriched["analysis_component"].isin(
            ["Company semantic profile", "Company semantic diversity"]
        )
    )
    enriched.loc[wp3_company_summary, "semantic_analysis_component"] = (
        "WP3 frozen company-level metric"
    )

    enriched["semantic_evidence_key"] = enriched.apply(
        lambda row: stable_id(
            "SEMANTIC_EVIDENCE",
            row["company"],
            row["work_package"],
            row["semantic_analysis_component"],
            row["claim_category"],
            row["originating_claim"],
            row["metric"],
            "" if pd.isna(row["value"]) else f"{safe_numeric(row['value']):.12g}"
            if np.isfinite(safe_numeric(row["value"]))
            else row["value"],
            row["comparison"],
            row["direction"],
            row["statistical_status"],
            row["evidence_status"],
        ),
        axis=1,
    )

    enriched["semantic_duplicate_count"] = enriched.groupby(
        "semantic_evidence_key", dropna=False
    )["evidence_id"].transform("size")
    enriched["is_semantic_duplicate"] = enriched["semantic_duplicate_count"].gt(1)

    return enriched


# =============================================================================
# Network construction
# =============================================================================


def build_nodes(
    claims: pd.DataFrame,
    ledger: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for company in COMPANIES:
        rows.append(
            {
                "node_id": f"COMPANY::{company}",
                "node_type": "company",
                "label": company,
                "company": company,
                "work_package": "",
                "claim_id": "",
                "evidence_id": "",
                "source_file": "",
                "status": "",
                "is_mapped_claim": "",
                "metadata_json": json_dumps_compact({}),
            }
        )

    work_packages = sorted(set(ledger["work_package"].dropna().astype(str)))
    for work_package in work_packages:
        rows.append(
            {
                "node_id": f"WORK_PACKAGE::{work_package}",
                "node_type": "work_package",
                "label": work_package,
                "company": "",
                "work_package": work_package,
                "claim_id": "",
                "evidence_id": "",
                "source_file": "",
                "status": "",
                "is_mapped_claim": "",
                "metadata_json": json_dumps_compact({}),
            }
        )

    for _, claim in claims.iterrows():
        rows.append(
            {
                "node_id": claim["claim_node_id"],
                "node_type": "claim",
                "label": claim["claim_text"],
                "company": claim["company"],
                "work_package": "WP8",
                "claim_id": claim["claim_id"],
                "evidence_id": "",
                "source_file": "",
                "status": claim["overall_consistency"],
                "is_mapped_claim": bool(claim["is_mapped_claim"]),
                "metadata_json": json_dumps_compact(
                    {
                        "claim_category": claim["claim_category"],
                        "overall_consistency": claim["overall_consistency"],
                    }
                ),
            }
        )

    for _, evidence in ledger.iterrows():
        rows.append(
            {
                "node_id": evidence["evidence_node_id"],
                "node_type": "evidence",
                "label": evidence["evidence_statement"],
                "company": evidence["company"],
                "work_package": evidence["work_package"],
                "claim_id": evidence["claim_id"],
                "evidence_id": evidence["evidence_id"],
                "source_file": evidence["source_file"],
                "status": evidence["evidence_status"],
                "is_mapped_claim": bool(evidence["is_mapped_claim"]),
                "metadata_json": json_dumps_compact(
                    {
                        "analysis_component": normalise_text(evidence["analysis_component"]),
                        "metric": normalise_text(evidence["metric"]),
                        "value": None
                        if pd.isna(evidence["value"])
                        else normalise_text(evidence["value"]),
                        "comparison": normalise_text(evidence["comparison"]),
                        "direction": normalise_text(evidence["direction"]),
                        "statistical_status": normalise_text(evidence["statistical_status"]),
                        "evidence_relevance": normalise_text(evidence["evidence_relevance"]),
                        "semantic_evidence_key": evidence["semantic_evidence_key"],
                        "semantic_duplicate_count": int(evidence["semantic_duplicate_count"]),
                    }
                ),
            }
        )

    source_groups = ledger.groupby(["source_node_id", "source_file"], dropna=False)
    for (source_node_id, source_file), group in source_groups:
        rows.append(
            {
                "node_id": source_node_id,
                "node_type": "source_file",
                "label": source_file,
                "company": pipe_join(group["company"]),
                "work_package": pipe_join(group["work_package"]),
                "claim_id": "",
                "evidence_id": "",
                "source_file": source_file,
                "status": "",
                "is_mapped_claim": "",
                "metadata_json": json_dumps_compact(
                    {
                        "n_evidence_records": int(len(group)),
                        "n_claims": int(group["claim_id"].nunique()),
                    }
                ),
            }
        )

    nodes = pd.DataFrame(rows)
    if nodes["node_id"].duplicated().any():
        duplicates = nodes.loc[nodes["node_id"].duplicated(keep=False)]
        raise RuntimeError(
            "Traceability node IDs are not unique:\n"
            + duplicates[["node_id", "node_type", "label"]].to_string(index=False)
        )

    node_order = pd.CategoricalDtype(
        ["company", "claim", "work_package", "evidence", "source_file"],
        ordered=True,
    )
    nodes["node_type"] = nodes["node_type"].astype(node_order)
    nodes = nodes.sort_values(["node_type", "company", "work_package", "node_id"])
    nodes["node_type"] = nodes["node_type"].astype(str)
    return nodes.reset_index(drop=True)


def build_edges(
    claims: pd.DataFrame,
    ledger: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add_edge(
        source_node_id: str,
        target_node_id: str,
        edge_type: str,
        company: str = "",
        work_package: str = "",
        claim_id: str = "",
        evidence_id: str = "",
        evidence_status: str = "",
        source_file: str = "",
        metadata: dict[str, object] | None = None,
    ) -> None:
        edge_id = stable_id(
            "EDGE",
            source_node_id,
            target_node_id,
            edge_type,
            evidence_id,
            source_file,
        )
        rows.append(
            {
                "edge_id": edge_id,
                "source_node_id": source_node_id,
                "target_node_id": target_node_id,
                "edge_type": edge_type,
                "company": company,
                "work_package": work_package,
                "claim_id": claim_id,
                "evidence_id": evidence_id,
                "evidence_status": evidence_status,
                "source_file": source_file,
                "metadata_json": json_dumps_compact(metadata or {}),
            }
        )

    for _, claim in claims.iterrows():
        add_edge(
            source_node_id=claim["company_node_id"],
            target_node_id=claim["claim_node_id"],
            edge_type="company_has_claim",
            company=claim["company"],
            work_package="WP8",
            claim_id=claim["claim_id"],
            metadata={"claim_category": claim["claim_category"]},
        )

    for _, evidence in ledger.iterrows():
        add_edge(
            source_node_id=evidence["work_package_node_id"],
            target_node_id=evidence["evidence_node_id"],
            edge_type="work_package_produces_evidence",
            company=evidence["company"],
            work_package=evidence["work_package"],
            claim_id=evidence["claim_id"],
            evidence_id=evidence["evidence_id"],
            evidence_status=evidence["evidence_status"],
            source_file=evidence["source_file"],
            metadata={"analysis_component": evidence["analysis_component"]},
        )
        add_edge(
            source_node_id=evidence["evidence_node_id"],
            target_node_id=evidence["claim_node_id"],
            edge_type=f"evidence_{evidence['evidence_status']}_claim",
            company=evidence["company"],
            work_package=evidence["work_package"],
            claim_id=evidence["claim_id"],
            evidence_id=evidence["evidence_id"],
            evidence_status=evidence["evidence_status"],
            source_file=evidence["source_file"],
            metadata={
                "evidence_relevance": evidence["evidence_relevance"],
                "semantic_evidence_key": evidence["semantic_evidence_key"],
            },
        )
        add_edge(
            source_node_id=evidence["source_node_id"],
            target_node_id=evidence["evidence_node_id"],
            edge_type="source_file_contains_evidence",
            company=evidence["company"],
            work_package=evidence["work_package"],
            claim_id=evidence["claim_id"],
            evidence_id=evidence["evidence_id"],
            evidence_status=evidence["evidence_status"],
            source_file=evidence["source_file"],
            metadata={"source_row": evidence["source_row"]},
        )

    edges = pd.DataFrame(rows)
    if edges["edge_id"].duplicated().any():
        duplicates = edges.loc[edges["edge_id"].duplicated(keep=False)]
        raise RuntimeError(
            "Traceability edge IDs are not unique:\n"
            + duplicates[["edge_id", "source_node_id", "target_node_id", "edge_type"]]
            .head(20)
            .to_string(index=False)
        )

    return edges.sort_values(
        ["company", "claim_id", "work_package", "edge_type", "evidence_id"]
    ).reset_index(drop=True)


# =============================================================================
# Traceability tables
# =============================================================================


def build_claim_evidence_traceability(ledger: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "claim_id",
        "claim_node_id",
        "company",
        "claim_category",
        "originating_claim",
        "is_mapped_claim",
        "overall_consistency",
        "evidence_id",
        "evidence_node_id",
        "work_package",
        "analysis_component",
        "evidence_statement",
        "metric",
        "value",
        "comparison",
        "direction",
        "statistical_status",
        "evidence_status",
        "evidence_relevance",
        "rationale",
        "source_file",
        "source_row",
        "source_node_id",
        "semantic_evidence_key",
        "semantic_duplicate_count",
        "is_semantic_duplicate",
    ]
    return ledger[columns].sort_values(
        ["company", "claim_id", "work_package", "evidence_status", "evidence_id"]
    )


def build_claim_traceability_summary(
    claims: pd.DataFrame,
    ledger: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for _, claim in claims.iterrows():
        subset = ledger[ledger["claim_id"] == claim["claim_id"]]
        empirical = subset[subset["work_package"].isin(EMPIRICAL_WORK_PACKAGES)]
        direct = empirical[empirical["evidence_relevance"].eq("Primary")]

        counts = empirical["evidence_status"].value_counts()
        represented = sorted(set(empirical["work_package"].dropna().astype(str)))
        direct_represented = sorted(set(direct["work_package"].dropna().astype(str)))

        rows.append(
            {
                "claim_id": claim["claim_id"],
                "company": claim["company"],
                "claim_category": claim["claim_category"],
                "claim_text": claim["claim_text"],
                "is_mapped_claim": bool(claim["is_mapped_claim"]),
                "overall_consistency": claim["overall_consistency"],
                "n_evidence_records": int(len(subset)),
                "n_empirical_evidence_records": int(len(empirical)),
                "n_unique_semantic_evidence": int(
                    empirical["semantic_evidence_key"].nunique()
                ),
                "n_primary_evidence_records": int(len(direct)),
                "n_supporting": int(counts.get("supporting", 0)),
                "n_conflicting": int(counts.get("conflicting", 0)),
                "n_qualifying": int(counts.get("qualifying", 0)),
                "n_unavailable": int(counts.get("unavailable", 0)),
                "n_work_packages_represented": int(len(represented)),
                "work_packages_represented": "|".join(represented),
                "n_primary_work_packages_represented": int(len(direct_represented)),
                "primary_work_packages_represented": "|".join(direct_represented),
                "empirical_workpackage_coverage": len(represented)
                / len(EMPIRICAL_WORK_PACKAGES),
                "n_source_files": int(empirical["source_file"].nunique()),
                "source_files": pipe_join(empirical["source_file"]),
                "has_supporting_path": bool((empirical["evidence_status"] == "supporting").any()),
                "has_conflicting_path": bool((empirical["evidence_status"] == "conflicting").any()),
                "has_complete_source_path": bool(
                    empirical["source_file"].map(normalise_text).ne("").all()
                    if len(empirical) > 0
                    else False
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["company", "claim_category", "claim_id"]
    ).reset_index(drop=True)


def build_workpackage_claim_coverage(
    claims: pd.DataFrame,
    ledger: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for _, claim in claims.iterrows():
        for work_package in EMPIRICAL_WORK_PACKAGES:
            subset = ledger[
                (ledger["claim_id"] == claim["claim_id"])
                & (ledger["work_package"] == work_package)
            ]
            counts = subset["evidence_status"].value_counts()
            rows.append(
                {
                    "claim_id": claim["claim_id"],
                    "company": claim["company"],
                    "claim_category": claim["claim_category"],
                    "claim_text": claim["claim_text"],
                    "is_mapped_claim": bool(claim["is_mapped_claim"]),
                    "work_package": work_package,
                    "is_represented": bool(len(subset) > 0),
                    "n_evidence_records": int(len(subset)),
                    "n_unique_semantic_evidence": int(
                        subset["semantic_evidence_key"].nunique()
                    ),
                    "n_supporting": int(counts.get("supporting", 0)),
                    "n_conflicting": int(counts.get("conflicting", 0)),
                    "n_qualifying": int(counts.get("qualifying", 0)),
                    "n_unavailable": int(counts.get("unavailable", 0)),
                    "n_source_files": int(subset["source_file"].nunique()),
                    "source_files": pipe_join(subset["source_file"]),
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["company", "claim_id", "work_package"]
    ).reset_index(drop=True)


def build_source_file_traceability(ledger: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source_file, group in ledger.groupby("source_file", dropna=False, sort=True):
        counts = group["evidence_status"].value_counts()
        rows.append(
            {
                "source_node_id": group["source_node_id"].iloc[0],
                "source_file": source_file,
                "work_packages": pipe_join(group["work_package"]),
                "companies": pipe_join(group["company"]),
                "n_claims": int(group["claim_id"].nunique()),
                "claim_ids": pipe_join(group["claim_id"]),
                "n_evidence_records": int(len(group)),
                "n_unique_semantic_evidence": int(
                    group["semantic_evidence_key"].nunique()
                ),
                "n_supporting": int(counts.get("supporting", 0)),
                "n_conflicting": int(counts.get("conflicting", 0)),
                "n_qualifying": int(counts.get("qualifying", 0)),
                "n_unavailable": int(counts.get("unavailable", 0)),
                "n_semantic_duplicate_records": int(
                    group["is_semantic_duplicate"].sum()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["work_packages", "source_file"])


def build_orphan_report(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    claims: pd.DataFrame,
    ledger: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    node_ids = set(nodes["node_id"])
    edge_sources = set(edges["source_node_id"])
    edge_targets = set(edges["target_node_id"])
    connected = edge_sources | edge_targets

    for _, node in nodes.iterrows():
        if node["node_id"] not in connected:
            rows.append(
                {
                    "orphan_type": "isolated_node",
                    "record_id": node["node_id"],
                    "record_type": node["node_type"],
                    "company": node["company"],
                    "claim_id": node["claim_id"],
                    "evidence_id": node["evidence_id"],
                    "detail": "Node has no incoming or outgoing traceability edge.",
                }
            )

    for _, edge in edges.iterrows():
        if edge["source_node_id"] not in node_ids:
            rows.append(
                {
                    "orphan_type": "missing_edge_source",
                    "record_id": edge["edge_id"],
                    "record_type": edge["edge_type"],
                    "company": edge["company"],
                    "claim_id": edge["claim_id"],
                    "evidence_id": edge["evidence_id"],
                    "detail": f"Missing source node: {edge['source_node_id']}",
                }
            )
        if edge["target_node_id"] not in node_ids:
            rows.append(
                {
                    "orphan_type": "missing_edge_target",
                    "record_id": edge["edge_id"],
                    "record_type": edge["edge_type"],
                    "company": edge["company"],
                    "claim_id": edge["claim_id"],
                    "evidence_id": edge["evidence_id"],
                    "detail": f"Missing target node: {edge['target_node_id']}",
                }
            )

    for _, claim in claims.iterrows():
        subset = ledger[ledger["claim_id"] == claim["claim_id"]]
        if subset.empty:
            rows.append(
                {
                    "orphan_type": "claim_without_evidence",
                    "record_id": claim["claim_node_id"],
                    "record_type": "claim",
                    "company": claim["company"],
                    "claim_id": claim["claim_id"],
                    "evidence_id": "",
                    "detail": "WP8 claim has no WP9a evidence record.",
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "orphan_type",
                "record_id",
                "record_type",
                "company",
                "claim_id",
                "evidence_id",
                "detail",
            ]
        )

    return pd.DataFrame(rows).sort_values(
        ["orphan_type", "company", "claim_id", "record_id"]
    )


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = resolve_inputs()
    input_manifest = build_input_manifest(paths)
    data = load_inputs(paths)

    claims = build_claim_table(data["ledger"], data["consistency"])
    ledger = enrich_ledger_with_claim_ids(data["ledger"], claims)

    # Add company node IDs to claims after canonical company validation.
    claims["company_node_id"] = claims["company"].map(
        lambda value: f"COMPANY::{value}"
    )

    nodes = build_nodes(claims, ledger)
    edges = build_edges(claims, ledger)
    traceability = build_claim_evidence_traceability(ledger)
    claim_summary = build_claim_traceability_summary(claims, ledger)
    coverage = build_workpackage_claim_coverage(claims, ledger)
    source_summary = build_source_file_traceability(ledger)
    orphans = build_orphan_report(nodes, edges, claims, ledger)

    # Deterministic integrity checks.
    n_claims_without_evidence = int(
        claim_summary["n_evidence_records"].eq(0).sum()
    )
    n_broken_edges = int(
        orphans["orphan_type"].isin(
            ["missing_edge_source", "missing_edge_target"]
        ).sum()
        if not orphans.empty
        else 0
    )

    run_summary = pd.DataFrame(
        [
            {
                "analysis": "WP9b Evidence Traceability Network",
                "n_companies": int(claims["company"].nunique()),
                "n_claims": int(len(claims)),
                "n_mapped_claims": int(claims["is_mapped_claim"].sum()),
                "n_unmapped_claims": int((~claims["is_mapped_claim"]).sum()),
                "n_evidence_records": int(len(ledger)),
                "n_unique_semantic_evidence": int(
                    ledger["semantic_evidence_key"].nunique()
                ),
                "n_source_files": int(ledger["source_file"].nunique()),
                "n_network_nodes": int(len(nodes)),
                "n_network_edges": int(len(edges)),
                "n_company_nodes": int((nodes["node_type"] == "company").sum()),
                "n_claim_nodes": int((nodes["node_type"] == "claim").sum()),
                "n_work_package_nodes": int(
                    (nodes["node_type"] == "work_package").sum()
                ),
                "n_evidence_nodes": int((nodes["node_type"] == "evidence").sum()),
                "n_source_file_nodes": int(
                    (nodes["node_type"] == "source_file").sum()
                ),
                "n_claims_without_evidence": n_claims_without_evidence,
                "n_broken_edges": n_broken_edges,
                "n_orphan_records": int(len(orphans)),
                "all_edges_resolve": bool(n_broken_edges == 0),
                "all_claims_traceable": bool(n_claims_without_evidence == 0),
                "output_directory": str(OUTPUT_DIR.relative_to(ROOT)),
            }
        ]
    )

    input_manifest.to_csv(
        OUTPUT_DIR / "traceability_input_manifest.csv", index=False
    )
    nodes.to_csv(OUTPUT_DIR / "traceability_nodes.csv", index=False)
    edges.to_csv(OUTPUT_DIR / "traceability_edges.csv", index=False)
    traceability.to_csv(
        OUTPUT_DIR / "claim_evidence_traceability.csv", index=False
    )
    claim_summary.to_csv(
        OUTPUT_DIR / "claim_traceability_summary.csv", index=False
    )
    coverage.to_csv(
        OUTPUT_DIR / "workpackage_claim_coverage.csv", index=False
    )
    source_summary.to_csv(
        OUTPUT_DIR / "source_file_traceability.csv", index=False
    )
    orphans.to_csv(OUTPUT_DIR / "traceability_orphans.csv", index=False)
    run_summary.to_csv(
        OUTPUT_DIR / "traceability_run_summary.csv", index=False
    )

    print("=" * 88)
    print("WP9b EVIDENCE TRACEABILITY NETWORK COMPLETE")
    print("=" * 88)
    print(f"Project root: {ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"WP8 claims represented: {len(claims):,}")
    print(f"Mapped claims: {claims['is_mapped_claim'].sum():,}")
    print(f"Evidence records linked: {len(ledger):,}")
    print(f"Unique semantic evidence records: {ledger['semantic_evidence_key'].nunique():,}")
    print(f"Source files represented: {ledger['source_file'].nunique():,}")
    print(f"Traceability nodes: {len(nodes):,}")
    print(f"Traceability edges: {len(edges):,}")
    print(f"Claims without evidence: {n_claims_without_evidence:,}")
    print(f"Broken edges: {n_broken_edges:,}")
    print(f"Orphan records: {len(orphans):,}")
    print("\nOutputs:")
    for filename in (
        "traceability_nodes.csv",
        "traceability_edges.csv",
        "claim_evidence_traceability.csv",
        "claim_traceability_summary.csv",
        "workpackage_claim_coverage.csv",
        "source_file_traceability.csv",
        "traceability_orphans.csv",
        "traceability_input_manifest.csv",
        "traceability_run_summary.csv",
    ):
        print(f"  - {OUTPUT_DIR / filename}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\nWP9b FAILED", file=sys.stderr)
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise
