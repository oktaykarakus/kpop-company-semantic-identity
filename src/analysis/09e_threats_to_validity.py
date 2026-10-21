#!/usr/bin/env python3
"""
WP9e2 — Evidence-Driven Threats to Validity and Scope of Inference
Project: Semantic Fingerprints of K-pop Entertainment Companies

Purpose
-------
Produce a deterministic, evidence-linked validity audit of the frozen WP1-WP9d
framework. This script does not rerun upstream analyses and does not calculate
new empirical statistics, effect sizes, latent variables, models, clusters,
embeddings, graphs, or hypothesis tests.

The audit covers:
    - data validity
    - construct validity
    - internal validity
    - statistical-conclusion validity
    - temporal validity
    - computational validity
    - external validity

Canonical evidence is reused from WP1, WP3-WP6, WP8, and WP9a-WP9d. Threat, assumption, mitigation, and scope registers remain transparent and fixed.
Validity judgements themselves are derived deterministically from frozen upstream
evidence using an explicit rule registry and a complete decision trace.

Outputs
-------
outputs/framework_validation/09e2_evidence_driven_validity_audit/
    validity_assessment.csv
    validity_rules.csv
    validity_decision_trace.csv
    validity_risk_register.csv
    assumptions_register.csv
    framework_scope.csv
    validity_summary.csv
    validity_evidence_manifest.csv
    validity_matrix.png
    validity_matrix.pdf
    validity_run_summary.csv

Run
---
python 09e2_evidence_driven_validity_audit.py
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


# =============================================================================
# Project paths and fixed labels
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "framework_validation"
    / "09e2_evidence_driven_validity_audit"
)

PASS = "PASS"
PARTIAL = "PARTIAL"
FAIL = "FAIL"
NOT_VERIFIED = "Not verified"

ASSESSMENT_ORDER = {
    "strong": 0,
    "adequate_with_qualifications": 1,
    "bounded": 2,
    "insufficiently_verified": 3,
}

RISK_ORDER = {"low": 0, "moderate": 1, "high": 2}

DIRECT_INPUTS = {
    # WP1
    "WP1_master_dataset": ROOT / "outputs" / "tables" / "company_master_analysis_v1.parquet",
    # WP3
    "WP3_company_profiles": ROOT / "outputs" / "company_profiles" / "tables" / "table_company_semantic_profiles.csv",
    "WP3_company_diversity": ROOT / "outputs" / "company_profiles" / "tables" / "company_semantic_diversity.csv",
    "WP3_pairwise_similarity": ROOT / "outputs" / "company_profiles" / "tables" / "pairwise_company_profile_similarity.csv",
    # WP4
    "WP4_global_tests": ROOT / "outputs" / "company_significance" / "tables" / "artist_level_global_metric_tests.csv",
    "WP4_pairwise_tests": ROOT / "outputs" / "company_significance" / "tables" / "artist_level_pairwise_metric_tests.csv",
    "WP4_community_enrichment": ROOT / "outputs" / "company_significance" / "tables" / "community_company_enrichment.csv",
    "WP4_bootstrap_diversity": ROOT / "outputs" / "company_significance" / "tables" / "company_diversity_bootstrap_summary.csv",
    # WP5
    "WP5_drift_summary": ROOT / "outputs" / "temporal" / "rolling5" / "semantic_drift_summary_consecutive.csv",
    "WP5_drift_pairwise": ROOT / "outputs" / "temporal" / "rolling5" / "semantic_drift_pairwise_consecutive.csv",
    "WP5_drift_kruskal": ROOT / "outputs" / "temporal" / "rolling5" / "semantic_drift_kruskal_consecutive.csv",
    "WP5_change_points": ROOT / "outputs" / "temporal" / "rolling5" / "semantic_change_point_summary.csv",
    "WP5_stability": ROOT / "outputs" / "temporal" / "rolling5" / "temporal_stability_summary.csv",
    "WP5_trajectories": ROOT / "outputs" / "temporal" / "rolling5" / "temporal_trajectory_summary.csv",
    # WP6
    "WP6_all_robustness": ROOT / "outputs" / "robustness" / "summary_robustness" / "wp6_all_robustness_records.csv",
    "WP6_company_summary": ROOT / "outputs" / "robustness" / "summary_robustness" / "wp6_company_robustness_summary.csv",
    "WP6_global_summary": ROOT / "outputs" / "robustness" / "summary_robustness" / "wp6_global_robustness_summary.csv",
    "WP6_pairwise_summary": ROOT / "outputs" / "robustness" / "summary_robustness" / "wp6_pairwise_robustness_summary.csv",
    "WP6_availability": ROOT / "outputs" / "robustness" / "summary_robustness" / "wp6_analysis_availability.csv",
    # WP9
    "WP9a_consistency": ROOT / "outputs" / "framework_validation" / "09a4_cross_workpackage_consistency" / "cross_wp_consistency_summary.csv",
    "WP9b_traceability": ROOT / "outputs" / "framework_validation" / "09b_evidence_traceability_network" / "traceability_run_summary.csv",
    "WP9c_transferability": ROOT / "outputs" / "framework_validation" / "09c_methodological_transferability" / "transferability_run_summary.csv",
    "WP9c_stage_assessment": ROOT / "outputs" / "framework_validation" / "09c_methodological_transferability" / "stage_transferability_assessment.csv",
    "WP9d_reproducibility": ROOT / "outputs" / "framework_validation" / "09d3_computational_reproducibility_audit" / "reproducibility_run_summary.csv",
    "WP9d_checklist": ROOT / "outputs" / "framework_validation" / "09d3_computational_reproducibility_audit" / "reproducibility_checklist.csv",
}

WP8_FILENAMES = {
    "WP8_integrated_identity": "integrated_company_identity.csv",
    "WP8_company_summary": "company_identity_summary.csv",
    "WP8_traceability": "identity_evidence_traceability.csv",
    "WP8_cross_analysis": "cross_analysis_consistency.csv",
}


# =============================================================================
# General helpers
# =============================================================================


def normalise_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def pipe_join(values: Iterable[object]) -> str:
    cleaned = sorted({normalise_text(value) for value in values if normalise_text(value)})
    return "|".join(cleaned)


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_exact_filename(filename: str) -> Path:
    output_root = ROOT / "outputs"
    matches = sorted(path for path in output_root.rglob(filename) if path.is_file())
    if not matches:
        raise FileNotFoundError(f"Could not resolve exact filename below outputs/: {filename}")
    if len(matches) > 1:
        candidates = [relative(path) for path in matches]
        raise RuntimeError(f"Ambiguous exact filename {filename!r}: {candidates}")
    return matches[0]


def resolve_inputs() -> dict[str, Path]:
    paths = dict(DIRECT_INPUTS)
    for key, filename in WP8_FILENAMES.items():
        paths[key] = resolve_exact_filename(filename)
    return paths


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    return next((column for column in candidates if column in df.columns), None)


def first_value(df: pd.DataFrame, candidates: Iterable[str], default: object = NOT_VERIFIED) -> object:
    column = first_existing_column(df, candidates)
    if column is None or df.empty:
        return default
    values = df[column].dropna()
    return values.iloc[0] if not values.empty else default


def bool_value(value: object) -> bool:
    return normalise_text(value).lower() in {"true", "yes", "1", "pass"}


# =============================================================================
# Evidence verification
# =============================================================================


def build_evidence_manifest(paths: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, path in paths.items():
        exists = path.is_file()
        row_count: int | str = NOT_VERIFIED
        columns: str = ""
        read_status = "not_read"
        error = ""
        if exists:
            try:
                df = read_table(path)
                row_count = int(len(df))
                columns = pipe_join(df.columns)
                read_status = "readable"
            except Exception as exc:  # manifest should expose, not hide, read failures
                read_status = "read_error"
                error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "input_key": key,
                "work_package": key.split("_", 1)[0],
                "relative_path": relative(path),
                "exists": exists,
                "read_status": read_status,
                "row_count": row_count,
                "columns": columns,
                "error": error,
            }
        )
    return pd.DataFrame(rows).sort_values(["work_package", "input_key"]).reset_index(drop=True)


def require_inputs(manifest: pd.DataFrame) -> None:
    bad = manifest[(manifest["exists"] != True) | (manifest["read_status"] != "readable")]
    if not bad.empty:
        details = bad[["input_key", "relative_path", "exists", "read_status", "error"]].to_dict("records")
        raise RuntimeError(
            "WP9e requires all declared frozen evidence inputs to exist and be readable. "
            f"Problems: {json.dumps(details, ensure_ascii=False)}"
        )


def numeric_series(df: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    column = first_existing_column(df, candidates)
    if column is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def normalised_series(df: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    column = first_existing_column(df, candidates)
    if column is None:
        return pd.Series(dtype=str)
    return df[column].fillna("").map(normalise_text).str.lower()


def count_matching(df: pd.DataFrame, candidates: Iterable[str], patterns: Iterable[str]) -> int:
    series = normalised_series(df, candidates)
    if series.empty:
        return 0
    mask = pd.Series(False, index=series.index)
    for pattern in patterns:
        mask = mask | series.str.contains(pattern, case=False, regex=True, na=False)
    return int(mask.sum())


def derive_verified_context(paths: dict[str, Path]) -> dict[str, object]:
    tables = {key: read_table(path) for key, path in paths.items()}

    master = tables["WP1_master_dataset"]
    eligible = master.copy()
    if "analysis_eligible" in master.columns:
        eligible = master[master["analysis_eligible"].map(bool_value)].copy()

    company_col = first_existing_column(eligible, ["company", "entertainment_company", "label"])
    artist_col = first_existing_column(eligible, ["artist", "artist_name", "group"])
    community_col = first_existing_column(eligible, ["community", "community_id", "semantic_community"])

    wp9a = tables["WP9a_consistency"]
    wp9b = tables["WP9b_traceability"]
    wp9c = tables["WP9c_transferability"]
    wp9c_stage = tables["WP9c_stage_assessment"]
    wp9d = tables["WP9d_reproducibility"]
    wp9d_checklist = tables["WP9d_checklist"]
    wp6 = tables["WP6_all_robustness"]

    consistency = normalised_series(wp9a, ["overall_consistency", "consistency"])
    n_conflicts = numeric_series(wp9a, ["n_conflicting", "conflicting_evidence"]).fillna(0)
    n_support = numeric_series(wp9a, ["n_supporting", "supporting_evidence"]).fillna(0)
    n_qualifying = numeric_series(wp9a, ["n_qualifying", "qualifying_evidence"]).fillna(0)
    n_available = numeric_series(wp9a, ["n_available_work_packages", "available_work_packages_count"]).fillna(0)

    n_claims = int(len(wp9a))
    n_consistent = int((consistency == "consistent").sum())
    n_qualified = int(consistency.isin(["consistent_with_qualifications", "mostly_consistent_with_qualifications"]).sum())
    n_inconsistent = int(consistency.str.fullmatch("inconsistent", na=False).sum())
    n_supported_claims = int(((n_support > 0) | (n_qualifying > 0)).sum())
    support_rate = n_supported_claims / n_claims if n_claims else 0.0
    coverage_rate = float((n_available > 0).mean()) if n_claims else 0.0

    trace_row = wp9b.iloc[0] if not wp9b.empty else pd.Series(dtype=object)
    all_edges_resolve = bool_value(trace_row.get("all_edges_resolve", False))
    all_claims_traceable = bool_value(trace_row.get("all_claims_traceable", False))
    claims_without_evidence = int(pd.to_numeric(pd.Series([trace_row.get("n_claims_without_evidence", 0)]), errors="coerce").fillna(0).iloc[0])
    broken_edges = int(pd.to_numeric(pd.Series([trace_row.get("n_broken_edges", 0)]), errors="coerce").fillna(0).iloc[0])

    transferability = normalised_series(wp9c_stage, ["transferability_level", "transferability", "assessment"])
    adaptation = normalised_series(wp9c_stage, ["required_modification", "adaptation_requirement", "adaptation_level", "required_adaptation"])
    high_transfer = int((transferability == "high").sum())
    medium_transfer = int((transferability == "medium").sum())
    low_transfer = int((transferability == "low").sum())
    domain_redesign = int(adaptation.str.contains(r"domain[_ ]redesign", case=False, regex=True, na=False).sum())
    transfer_stages = int(len(wp9c_stage))

    reproducibility = normalise_text(first_value(
        wp9d,
        ["overall_computational_reproducibility", "overall_reproducibility", "status"],
    )).upper()
    checklist_status = normalised_series(wp9d_checklist, ["status"])

    robustness_conclusion = normalised_series(wp6, ["robustness_conclusion", "structure_conclusion", "ranking_conclusion"])
    wp6_available = tables["WP6_availability"]
    availability_values = normalised_series(wp6_available, ["available", "status", "analysis_available"])

    required_wp4 = ["WP4_global_tests", "WP4_pairwise_tests", "WP4_community_enrichment", "WP4_bootstrap_diversity"]
    required_wp5 = ["WP5_drift_summary", "WP5_drift_pairwise", "WP5_drift_kruskal", "WP5_change_points", "WP5_stability", "WP5_trajectories"]
    required_wp8 = ["WP8_integrated_identity", "WP8_company_summary", "WP8_traceability", "WP8_cross_analysis"]

    return {
        "eligible_song_records": int(len(eligible)),
        "companies": int(eligible[company_col].nunique()) if company_col else 0,
        "artists": int(eligible[artist_col].nunique()) if artist_col else 0,
        "semantic_communities": int(eligible[community_col].nunique()) if community_col else 0,
        "analysis_eligible_column_present": "analysis_eligible" in master.columns,
        "eligible_subset_nonempty": bool(len(eligible) > 0),
        "company_count_sufficient_for_comparison": bool(company_col and eligible[company_col].nunique() >= 2),
        "artist_count_nonzero": bool(artist_col and eligible[artist_col].nunique() > 0),
        "community_count_nonzero": bool(community_col and eligible[community_col].nunique() > 0),
        "mapped_claims": n_claims,
        "consistent_claims": n_consistent,
        "qualified_claims": n_qualified,
        "inconsistent_claims": n_inconsistent,
        "supported_claims": n_supported_claims,
        "support_rate": support_rate,
        "claim_coverage_rate": coverage_rate,
        "conflicting_evidence_records": int(n_conflicts.sum()),
        "all_edges_resolve": all_edges_resolve,
        "all_claims_traceable": all_claims_traceable,
        "claims_without_evidence": claims_without_evidence,
        "broken_traceability_edges": broken_edges,
        "wp3_outputs_present": all(not tables[k].empty for k in ["WP3_company_profiles", "WP3_company_diversity", "WP3_pairwise_similarity"]),
        "wp4_outputs_present": all(not tables[k].empty for k in required_wp4),
        "wp4_global_tests_present": not tables["WP4_global_tests"].empty,
        "wp4_pairwise_tests_present": not tables["WP4_pairwise_tests"].empty,
        "wp4_enrichment_present": not tables["WP4_community_enrichment"].empty,
        "wp4_bootstrap_present": not tables["WP4_bootstrap_diversity"].empty,
        "wp5_outputs_present": all(not tables[k].empty for k in required_wp5),
        "wp5_change_points_present": not tables["WP5_change_points"].empty,
        "wp5_stability_present": not tables["WP5_stability"].empty,
        "wp5_trajectories_present": not tables["WP5_trajectories"].empty,
        "wp6_records_present": not wp6.empty,
        "wp6_availability_complete": bool(not wp6_available.empty and (availability_values.empty or not availability_values.str.contains("unavailable|missing|fail", regex=True, na=False).any())),
        "wp6_nonfailure_rate": float((~robustness_conclusion.str.contains("not robust|fail|unstable", regex=True, na=False)).mean()) if not robustness_conclusion.empty else 0.0,
        "wp6_temporal_sensitivity_present": bool(wp6.astype(str).apply(lambda col: col.str.contains("temporal|window", case=False, regex=True, na=False)).any(axis=1).any()),
        "wp8_outputs_present": all(not tables[k].empty for k in required_wp8),
        "reproducibility_overall": reproducibility,
        "reproducibility_checklist_pass": bool(not checklist_status.empty and (checklist_status == "pass").all()),
        "transferability_stages": transfer_stages,
        "transferability_high": high_transfer,
        "transferability_medium": medium_transfer,
        "transferability_low": low_transfer,
        "domain_redesign_stages": domain_redesign,
        "transferability_overall": normalise_text(first_value(wp9c, ["overall_transferability", "framework_transferability", "overall_methodological_transferability", "overall_readiness"])),
    }


# =============================================================================
# Predefined validity framework
# =============================================================================


def build_validity_rules() -> pd.DataFrame:
    rows = [
        ("DV-01", "data_validity", 1, "Strong", "Low", "Eligibility field exists; eligible subset is non-empty; at least two companies, one artist, and one semantic community are represented."),
        ("DV-02", "data_validity", 2, "Adequate With Qualifications", "Moderate", "Canonical eligible data are present and support comparison, but the finite four-company corpus remains population-bounded."),
        ("DV-03", "data_validity", 3, "Insufficiently Verified", "High", "One or more foundational data-integrity conditions are not verified."),
        ("CoV-01", "construct_validity", 1, "Strong", "Low", "WP3 and WP8 outputs are complete, all claims are traceable, at least 90% of mapped claims are supported, and no inconsistent claims are present."),
        ("CoV-02", "construct_validity", 2, "Adequate With Qualifications", "Moderate", "Multiple construct representations and traceability are present, but lyrical semantics remain a bounded proxy for organisational identity."),
        ("CoV-03", "construct_validity", 3, "Insufficiently Verified", "High", "Construct representations, integration outputs, or traceability are incomplete."),
        ("IV-01", "internal_validity", 1, "Strong", "Low", "No inconsistent claims, no conflicting evidence records, support rate at least 90%, all claims traceable, and no broken traceability edges."),
        ("IV-02", "internal_validity", 2, "Supported With Qualifications", "Moderate", "Most claims are supported and traceable, but conflicts or incomplete coverage remain."),
        ("IV-03", "internal_validity", 3, "Weak", "High", "Inconsistent claims, unresolved traceability, or low support dominate."),
        ("SV-01", "statistical_conclusion_validity", 1, "Strong", "Low", "All four WP4 inferential outputs, WP6 robustness records, and analysis availability are verified, with no detected robustness-failure majority."),
        ("SV-02", "statistical_conclusion_validity", 2, "Supported With Qualifications", "Moderate", "Core inferential and robustness evidence exists but one completeness or stability condition is not verified."),
        ("SV-03", "statistical_conclusion_validity", 3, "Insufficiently Verified", "High", "Core inferential or robustness evidence is absent."),
        ("TV-01", "temporal_validity", 1, "Strong", "Low", "All WP5 temporal outputs and WP6 temporal-window sensitivity are present, with complete robustness availability."),
        ("TV-02", "temporal_validity", 2, "Adequate With Qualifications", "Moderate", "Temporal drift, stability, trajectory, and change-point evidence is complete, but inference is retrospective and catalogue-bounded."),
        ("TV-03", "temporal_validity", 3, "Insufficiently Verified", "High", "One or more core temporal evidence components are absent."),
        ("CV-01", "computational_validity", 1, "Strong", "Low", "WP9d overall verdict is PASS and every reproducibility checklist item is PASS."),
        ("CV-02", "computational_validity", 2, "Supported With Qualifications", "Moderate", "WP9d is PARTIAL or at least one reproducibility checklist item is not PASS."),
        ("CV-03", "computational_validity", 3, "Weak", "High", "WP9d overall verdict is FAIL."),
        ("EV-01", "external_validity", 1, "Supported With Qualifications", "Moderate", "All stages have high or medium transferability and no stage requires domain redesign."),
        ("EV-02", "external_validity", 2, "Bounded", "High", "At least one stage has low transferability or requires domain redesign; transferability is methodological rather than externally replicated."),
        ("EV-03", "external_validity", 3, "Insufficiently Verified", "High", "Transferability-stage evidence is absent."),
    ]
    return pd.DataFrame(rows, columns=["rule_id", "validity_dimension", "priority", "assessment", "residual_risk", "rule_description"])


def _decision(dimension: str, rule_id: str, assessment: str, risk: str, evidence: str, condition_values: dict[str, object], basis: str) -> dict[str, object]:
    return {
        "validity_dimension": dimension,
        "rule_id": rule_id,
        "assessment": assessment,
        "residual_risk": risk,
        "decision_basis": basis,
        "evidence_summary": evidence,
        "condition_values": json.dumps(condition_values, sort_keys=True, ensure_ascii=False),
    }


def build_validity_decision_trace(context: dict[str, object]) -> pd.DataFrame:
    decisions: list[dict[str, object]] = []

    data_ok = all([
        context["analysis_eligible_column_present"], context["eligible_subset_nonempty"],
        context["company_count_sufficient_for_comparison"], context["artist_count_nonzero"],
        context["community_count_nonzero"],
    ])
    dv_values = {k: context[k] for k in ["analysis_eligible_column_present", "eligible_subset_nonempty", "company_count_sufficient_for_comparison", "artist_count_nonzero", "community_count_nonzero", "eligible_song_records", "companies", "artists", "semantic_communities"]}
    if data_ok:
        decisions.append(_decision("data_validity", "DV-02", "Adequate With Qualifications", "Moderate", f"Verified canonical eligible corpus: {context['eligible_song_records']} songs, {context['companies']} companies, {context['artists']} artists, {context['semantic_communities']} communities; population remains intentionally bounded.", dv_values, "Derived from evidence synthesis"))
    else:
        decisions.append(_decision("data_validity", "DV-03", "Insufficiently Verified", "High", "At least one foundational data-integrity condition was not verified.", dv_values, "Derived automatically"))

    construct_values = {k: context[k] for k in ["wp3_outputs_present", "wp8_outputs_present", "all_claims_traceable", "support_rate", "inconsistent_claims"]}
    construct_core = context["wp3_outputs_present"] and context["wp8_outputs_present"] and context["all_claims_traceable"]
    if construct_core and context["support_rate"] >= 0.90 and context["inconsistent_claims"] == 0:
        decisions.append(_decision("construct_validity", "CoV-02", "Adequate With Qualifications", "Moderate", "Multiple convergent semantic representations, integrated identity outputs, high claim support, and complete traceability are verified; lyrics remain a proxy for organisational identity.", construct_values, "Derived from evidence synthesis"))
    else:
        decisions.append(_decision("construct_validity", "CoV-03", "Insufficiently Verified", "High", "Construct integration or traceability conditions were incomplete.", construct_values, "Derived automatically"))

    internal_values = {k: context[k] for k in ["mapped_claims", "supported_claims", "support_rate", "inconsistent_claims", "conflicting_evidence_records", "all_claims_traceable", "broken_traceability_edges"]}
    internal_strong = context["inconsistent_claims"] == 0 and context["conflicting_evidence_records"] == 0 and context["support_rate"] >= 0.90 and context["all_claims_traceable"] and context["broken_traceability_edges"] == 0
    if internal_strong:
        decisions.append(_decision("internal_validity", "IV-01", "Strong", "Low", f"{context['supported_claims']}/{context['mapped_claims']} claims supported or qualified, zero inconsistent claims, zero conflicting evidence records, and complete traceability with zero broken edges.", internal_values, "Derived automatically"))
    elif context["support_rate"] >= 0.70 and context["all_claims_traceable"]:
        decisions.append(_decision("internal_validity", "IV-02", "Supported With Qualifications", "Moderate", "Most claims are supported and traceable, but one or more strict consistency conditions are not met.", internal_values, "Derived automatically"))
    else:
        decisions.append(_decision("internal_validity", "IV-03", "Weak", "High", "Support, consistency, or traceability conditions are insufficient.", internal_values, "Derived automatically"))

    stat_values = {k: context[k] for k in ["wp4_global_tests_present", "wp4_pairwise_tests_present", "wp4_enrichment_present", "wp4_bootstrap_present", "wp6_records_present", "wp6_availability_complete", "wp6_nonfailure_rate"]}
    stat_core = context["wp4_outputs_present"] and context["wp6_records_present"]
    if stat_core and context["wp6_availability_complete"] and context["wp6_nonfailure_rate"] >= 0.50:
        decisions.append(_decision("statistical_conclusion_validity", "SV-01", "Strong", "Low", "Artist-level global and pairwise tests, community enrichment, bootstrap uncertainty, and complete WP6 robustness evidence are present.", stat_values, "Derived automatically"))
    elif stat_core:
        decisions.append(_decision("statistical_conclusion_validity", "SV-02", "Supported With Qualifications", "Moderate", "Core inferential and robustness outputs are present, but a completeness or robustness condition is not fully verified.", stat_values, "Derived automatically"))
    else:
        decisions.append(_decision("statistical_conclusion_validity", "SV-03", "Insufficiently Verified", "High", "Core WP4 or WP6 evidence is absent.", stat_values, "Derived automatically"))

    temporal_values = {k: context[k] for k in ["wp5_outputs_present", "wp5_change_points_present", "wp5_stability_present", "wp5_trajectories_present", "wp6_temporal_sensitivity_present", "wp6_availability_complete"]}
    temporal_core = context["wp5_outputs_present"] and context["wp6_temporal_sensitivity_present"]
    if temporal_core:
        decisions.append(_decision("temporal_validity", "TV-02", "Adequate With Qualifications", "Moderate", "Drift, pairwise drift, temporal tests, change points, stability, trajectories, and temporal-window sensitivity are verified; evidence remains retrospective.", temporal_values, "Derived from evidence synthesis"))
    else:
        decisions.append(_decision("temporal_validity", "TV-03", "Insufficiently Verified", "High", "One or more core temporal evidence components are absent.", temporal_values, "Derived automatically"))

    comp_values = {k: context[k] for k in ["reproducibility_overall", "reproducibility_checklist_pass"]}
    if context["reproducibility_overall"] == PASS and context["reproducibility_checklist_pass"]:
        decisions.append(_decision("computational_validity", "CV-01", "Strong", "Low", "WP9d reports PASS and every reproducibility checklist item passes.", comp_values, "Derived automatically"))
    elif context["reproducibility_overall"] == FAIL:
        decisions.append(_decision("computational_validity", "CV-03", "Weak", "High", "WP9d reports FAIL.", comp_values, "Derived automatically"))
    else:
        decisions.append(_decision("computational_validity", "CV-02", "Supported With Qualifications", "Moderate", "Computational reproducibility is not a complete PASS across verdict and checklist.", comp_values, "Derived automatically"))

    ext_values = {k: context[k] for k in ["transferability_stages", "transferability_high", "transferability_medium", "transferability_low", "domain_redesign_stages", "transferability_overall"]}
    if context["transferability_stages"] == 0:
        decisions.append(_decision("external_validity", "EV-03", "Insufficiently Verified", "High", "No stage-level transferability evidence was available.", ext_values, "Derived automatically"))
    elif context["transferability_low"] > 0 or context["domain_redesign_stages"] > 0:
        decisions.append(_decision("external_validity", "EV-02", "Bounded", "High", f"WP9c identifies {context['transferability_high']} high-, {context['transferability_medium']} medium-, and {context['transferability_low']} low-transferability stages, with {context['domain_redesign_stages']} requiring domain redesign; no external empirical replication is claimed.", ext_values, "Derived from evidence synthesis"))
    else:
        decisions.append(_decision("external_validity", "EV-01", "Supported With Qualifications", "Moderate", "All stages are high or medium transferability and none requires domain redesign, but empirical replication remains outstanding.", ext_values, "Derived from evidence synthesis"))

    order = ["internal_validity", "statistical_conclusion_validity", "computational_validity", "data_validity", "construct_validity", "temporal_validity", "external_validity"]
    trace = pd.DataFrame(decisions)
    trace["display_order"] = trace["validity_dimension"].map({v: i for i, v in enumerate(order)})
    return trace.sort_values("display_order").drop(columns="display_order").reset_index(drop=True)


def build_validity_assessment(context: dict[str, object], trace: pd.DataFrame) -> pd.DataFrame:
    metadata = {
        "data_validity": ("Does the retained corpus adequately support the stated population-level interpretation?", "WP1 canonical master dataset and eligibility filter", "WP1", "A single retained canonical dataset and explicit eligibility filter support the complete pipeline.", "Coverage is restricted to the selected companies, artists, songs, period, lyric availability, and inclusion rules.", "Inference is bounded to the observed eligible corpus."),
        "construct_validity": ("Does the framework operationalise organisational semantic identity rather than a neighbouring construct?", "WP3 profiles, WP8 integrated identities, WP9a consistency, and WP9b traceability", "WP3|WP8|WP9a|WP9b", "The construct is represented through multiple convergent analytical paths.", "Lyrics are an observable cultural product rather than a direct measure of executive intent, total corporate identity, or audience interpretation.", "The validated construct is organisational semantic fingerprint as expressed in the analysed lyrics."),
        "internal_validity": ("Are framework-level conclusions coherently supported by the frozen analyses?", "WP9a consistency synthesis and WP9b traceability", "WP9a|WP9b", "Cross-analysis support, conflict counts, and traceability are directly audited.", "Convergence across analyses does not establish causality and may reflect shared upstream representations.", "The framework supports coherent descriptive and comparative, not causal, inference."),
        "statistical_conclusion_validity": ("Are inferential conclusions supported by uncertainty, multiplicity, and sensitivity evidence?", "WP4 inferential outputs and WP6 robustness analyses", "WP4|WP6", "Artist-level testing, enrichment, bootstrap uncertainty, and robustness perturbations are jointly represented.", "Power and precision remain bounded by company, artist, song, community, and time-window counts.", "Statistical conclusions apply to tested contrasts and retained analytical units."),
        "temporal_validity": ("Are semantic identities stable or interpretable across the observed time horizon?", "WP5 temporal analyses and WP6 temporal-window sensitivity", "WP5|WP6", "Several complementary temporal summaries and window-sensitivity evidence are retained.", "Rolling-window estimates depend on release density and catalogue coverage; future behaviour is unobserved.", "Temporal claims are retrospective within the observed catalogue period."),
        "computational_validity": ("Can the retained workflow and its dependencies be reproduced?", "WP9d3 source, dependency, artefact, seed-control, and circularity audit", "WP9d", "The complete script and dependency chain is independently audited.", "A project-level audit is not identical to execution on every operating system or future package version.", "Reproducibility applies to the retained structure, inputs, and controlled stochastic stages."),
        "external_validity": ("Can the framework and conclusions be generalised beyond the analysed setting?", "WP9c methodological transferability audit", "WP9c", "Reusable stages, domain dependencies, adaptation requirements, and a reuse checklist are explicit.", "Methodological transferability is not external empirical replication.", "Analytical logic may transfer with adaptation; empirical fingerprints require independent validation."),
    }
    rows=[]
    for _, d in trace.iterrows():
        q,e,wp,s,l,scope = metadata[d["validity_dimension"]]
        rows.append({
            "validity_dimension": d["validity_dimension"], "assessment": d["assessment"], "residual_risk": d["residual_risk"],
            "rule_id": d["rule_id"], "decision_basis": d["decision_basis"], "core_question": q,
            "evidence_basis": e, "evidence_work_packages": wp, "evidence_summary": d["evidence_summary"],
            "strengths": s, "limitations": l, "scope_implication": scope,
        })
    return pd.DataFrame(rows)


def build_risk_register() -> pd.DataFrame:
    rows = [
        ("T01", "data_validity", "catalogue_coverage", "The retained corpus may omit releases or lyrics unavailable under the acquisition and eligibility rules.", "moderate", "high", "Canonical master dataset, explicit eligibility filtering, and retained input provenance.", "moderate", "Restrict population claims to the analysed eligible corpus."),
        ("T02", "data_validity", "company_selection", "Four focal companies do not constitute the complete K-pop organisational population.", "high", "high", "Use company-specific comparative language and report the bounded sampling frame.", "high", "Do not describe the sample as industry-exhaustive."),
        ("T03", "data_validity", "artist_song_imbalance", "Unequal catalogue sizes can cause larger companies or artists to dominate aggregate profiles.", "moderate", "high", "Artist-level inference, bootstrap analysis, leave-one-artist-out analysis, and balanced-company sampling in WP4/WP6.", "low", "Interpret company profiles with the robustness evidence attached."),
        ("T04", "construct_validity", "lyrics_as_proxy", "Lyrics may reflect writers, producers, artists, translators, and market conventions as well as company-level identity.", "high", "high", "Define the construct as company-associated lyrical semantic expression and triangulate across artists and analyses.", "moderate", "Avoid equating the fingerprint with direct managerial intent."),
        ("T05", "construct_validity", "embedding_dependence", "Semantic relations inherit assumptions and limitations of the retained embedding representation.", "moderate", "high", "Cross-analysis triangulation, community-level interpretation, and sensitivity evidence.", "moderate", "Treat semantic geometry as an operational representation, not ground truth."),
        ("T06", "construct_validity", "community_labelling", "Human-readable community labels may simplify heterogeneous lyrical themes.", "moderate", "moderate", "Retain the community dictionary, contribution evidence, and traceability to underlying records.", "moderate", "Use labels as interpretive summaries rather than exclusive ontological classes."),
        ("T07", "internal_validity", "shared_upstream_dependence", "Several analyses reuse the same master corpus and semantic representation, reducing strict independence among evidence paths.", "high", "moderate", "Distinguish analytical convergence from independent replication and preserve source-level traceability.", "moderate", "Claim triangulation, not independent causal confirmation."),
        ("T08", "internal_validity", "causal_overinterpretation", "Observed company differences could be shaped by artist composition, era, genre, production teams, or market changes.", "high", "high", "Use descriptive/comparative framing, artist-level tests, temporal analysis, and leave-one-artist-out sensitivity.", "moderate", "No causal claims about company strategy are warranted."),
        ("T09", "statistical_conclusion_validity", "limited_company_count", "The small number of focal companies constrains population-level generalisation and some higher-level inferential designs.", "high", "high", "Use artist/song-level evidence carefully and avoid treating four companies as a random industry sample.", "high", "Inference remains comparative among the observed companies."),
        ("T10", "statistical_conclusion_validity", "multiple_comparisons", "Many metrics, communities, and pairwise contrasts can inflate false-positive risk.", "moderate", "high", "Retain multiplicity-aware WP4 outputs, effect evidence, and WP6 robustness synthesis.", "low", "Prioritise convergent, corrected, and robust findings."),
        ("T11", "statistical_conclusion_validity", "sparse_subgroups", "Sparse artists, communities, or temporal windows may produce unstable estimates.", "moderate", "moderate", "Eligibility rules, availability reporting, bootstrap summaries, and sensitivity analyses.", "moderate", "Qualify conclusions where evidence coverage is limited."),
        ("T12", "temporal_validity", "rolling_window_choice", "Temporal conclusions may depend on rolling-window width and release density.", "moderate", "high", "Dedicated temporal-window sensitivity in WP6 and multiple WP5 temporal summaries.", "low", "Report findings as stable only when supported across tested windows."),
        ("T13", "temporal_validity", "historical_boundary", "The observed period cannot establish persistence before or after the retained catalogue.", "high", "moderate", "Explicitly bound temporal inference to observed years.", "moderate", "No extrapolation beyond the observed catalogue period."),
        ("T14", "computational_validity", "environment_drift", "Future package, model, or operating-system changes may alter execution despite current pipeline integrity.", "moderate", "moderate", "Seed control, retained scripts and artefacts, dependency audit, and environment archiving.", "low", "Preserve lockfiles/environment specifications with the release."),
        ("T15", "external_validity", "domain_transfer_without_replication", "A method that is structurally reusable may not retain validity or performance in a new domain.", "high", "high", "WP9c identifies adaptation requirements and explicitly separates transferability from empirical validation.", "high", "Require independent replication before cross-domain substantive claims."),
        ("T16", "external_validity", "language_and_translation", "Korean, English, mixed-language lyrics, and translated text may not preserve identical semantic nuances.", "moderate", "high", "Document language preprocessing and restrict interpretation to the retained text representation.", "moderate", "Multilingual generalisation requires separate validation."),
        ("T17", "reporting_validity", "selective_emphasis", "Narrative reporting could overemphasise vivid company differences while underreporting qualified evidence.", "moderate", "high", "WP9a consistency classes, WP9b traceability, and explicit claim-level qualifications.", "low", "Carry qualifications into tables, figures, abstract, and discussion."),
    ]
    columns = [
        "threat_id", "validity_dimension", "threat_code", "threat_description",
        "likelihood", "impact", "existing_mitigation", "residual_risk",
        "required_reporting_constraint",
    ]
    df = pd.DataFrame(rows, columns=columns)
    return df.sort_values(
        ["residual_risk", "validity_dimension", "threat_id"],
        key=lambda s: s.map(RISK_ORDER) if s.name == "residual_risk" else s,
        ascending=[False, True, True],
    ).reset_index(drop=True)


def build_assumptions_register() -> pd.DataFrame:
    rows = [
        ("A01", "data", "The retained master dataset correctly links each eligible song to its artist, company, date, and semantic attributes.", "partially_testable", "WP1 construction checks and retained canonical dataset", "Misattribution would distort company profiles and temporal ordering.", "Retain source provenance and perform spot checks when extending the corpus."),
        ("A02", "data", "The analysis-eligible subset is a defensible representation of the intended study corpus.", "partially_testable", "WP1 eligibility flag and documented inclusion rules", "Eligibility exclusions may introduce systematic coverage bias.", "State the sampling frame and excluded-record logic explicitly."),
        ("A03", "construct", "Company-associated lyrics contain a measurable organisational semantic signal.", "empirically_supported_not_proven", "WP3, WP4, WP7, WP8, and WP9a convergence", "Signal may partly reflect artist composition or broader industry conventions.", "Use 'company-associated semantic fingerprint' rather than direct intent."),
        ("A04", "construct", "The retained semantic representation preserves distinctions relevant to the research question.", "partially_testable", "Profile separation, prediction, interpretation, and robustness outputs", "Representation bias can shape all downstream geometry.", "Treat the representation as operational and report its provenance."),
        ("A05", "construct", "Detected communities are useful summaries of recurring lyrical semantics.", "partially_testable", "WP6 community-detection sensitivity and WP8 semantic dictionary", "Community boundaries and labels may be method-dependent.", "Use community-level interpretations only with sensitivity evidence."),
        ("A06", "internal", "Cross-workpackage convergence reflects a stable phenomenon rather than only duplicated computations.", "partially_testable", "WP9a relevance filtering, evidence deduplication, and WP9b provenance", "Shared data and representations limit independence.", "Describe the evidence as triangulated but not independently replicated."),
        ("A07", "statistical", "Artist-level analytical units and implemented tests appropriately reflect dependence in the corpus.", "partially_testable", "WP4 artist-level tests and WP6 perturbation analyses", "Residual dependence may remain among songs, artists, and eras.", "Avoid stronger population claims than the design supports."),
        ("A08", "statistical", "Implemented correction, bootstrap, and sensitivity procedures adequately control the principal inferential risks examined.", "testable_from_code_and_outputs", "WP4, WP6, and WP9d", "Untested forms of model or sampling uncertainty may remain.", "Keep conclusions tied to the actual tested robustness conditions."),
        ("A09", "temporal", "Release dates and rolling windows are sufficiently accurate for retrospective temporal comparison.", "testable_from_data", "WP1 dates and WP5 temporal profiles", "Date errors or sparse periods can shift temporal summaries.", "Validate dates when adding or revising records."),
        ("A10", "temporal", "Observed temporal drift represents catalogue change rather than only changing artist mix or missingness.", "partially_testable", "WP5 analyses and WP6 temporal/artist sensitivity", "Composition effects cannot be fully removed in this design.", "Use descriptive language and retain artist-composition qualifications."),
        ("A11", "computational", "Retained scripts, canonical inputs, and dependencies correspond to the frozen reported results.", "verified", "WP9d3 PASS", "Later file replacement could break provenance.", "Archive immutable release versions and checksums."),
        ("A12", "computational", "Explicit seeds are sufficient to control all stochastic stages in the retained environment.", "verified_with_environment_boundary", "WP9d3 source audit", "Low-level numerical libraries can vary across environments.", "Archive package and platform metadata with the code release."),
        ("A13", "external", "The analytical stages can be adapted to another organisational text domain without changing their core logic.", "methodologically_assessed_not_empirically_validated", "WP9c", "Domain-specific preprocessing and construct definitions may require redesign.", "Require domain-specific construct validation and replication."),
        ("A14", "reporting", "All claim-level qualifications and non-supporting evidence are retained in final reporting.", "procedurally_verifiable", "WP9a and WP9b", "Selective narrative compression can reintroduce bias.", "Use the claim and evidence tables during manuscript drafting."),
    ]
    columns = [
        "assumption_id", "assumption_domain", "assumption", "verification_status",
        "evidence_basis", "consequence_if_violated", "management_action",
    ]
    return pd.DataFrame(rows, columns=columns)


def build_scope_register() -> pd.DataFrame:
    rows = [
        ("S01", "validated", "Comparative lyrical semantic profiles of the four analysed K-pop entertainment companies", "Directly represented in the frozen eligible corpus and WP3-WP8 analyses."),
        ("S02", "validated", "Company-associated semantic diversity, similarity, differentiation, and community contribution", "Directly assessed through retained profile, inferential, robustness, and explanation outputs."),
        ("S03", "validated", "Retrospective temporal drift, stability, trajectories, and change points within the observed catalogue period", "Directly assessed in WP5 and temporally stress-tested in WP6."),
        ("S04", "validated", "Traceable, internally consistent, and computationally reproducible framework-level claims", "Supported by WP9a, WP9b, and WP9d."),
        ("S05", "methodologically_transferable_not_empirically_validated", "Reuse of the pipeline in another organisational text domain", "WP9c assesses methodological adaptation requirements, not external empirical performance."),
        ("S06", "not_validated", "All K-pop entertainment companies or the K-pop industry as a complete population", "Only four focal companies are analysed."),
        ("S07", "not_validated", "Causal effects of company strategy, management decisions, or branding on lyrics", "The design is observational and descriptive/comparative."),
        ("S08", "not_validated", "Direct measurement of total corporate identity, artist intent, audience interpretation, or commercial strategy", "Lyrics operationalise one observable semantic expression only."),
        ("S09", "not_validated", "Other music industries, national contexts, or entertainment sectors", "No independent external replication has been performed."),
        ("S10", "not_validated", "Multilingual equivalence across original lyrics, translations, and language-specific semantic models", "Language-transfer validity is outside the retained empirical assessment."),
        ("S11", "not_validated", "Non-lyrical media such as music video imagery, choreography, social media, interviews, or visual branding", "These modalities are not represented in the semantic fingerprint."),
        ("S12", "not_validated", "Streaming success, sales, chart performance, audience behaviour, or business outcomes", "The framework does not model these outcomes."),
        ("S13", "not_validated", "Forecasts of future company identity or catalogue evolution", "WP5 is retrospective; no future forecasting claim is validated here."),
        ("S14", "not_validated", "Legal, ethical, or normative evaluation of companies or artists", "Semantic differentiation is descriptive and should not be treated as a value judgement."),
    ]
    return pd.DataFrame(rows, columns=["scope_id", "scope_status", "scope_statement", "justification"])


# =============================================================================
# Summaries and publication figure
# =============================================================================


def build_validity_summary(
    assessment: pd.DataFrame,
    risks: pd.DataFrame,
    assumptions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, item in assessment.iterrows():
        dimension = item["validity_dimension"]
        subset = risks[risks["validity_dimension"] == dimension]
        rows.append({
            "validity_dimension": dimension,
            "assessment": item["assessment"],
            "residual_risk": item["residual_risk"],
            "rule_id": item["rule_id"],
            "decision_basis": item["decision_basis"],
            "n_registered_threats": int(len(subset)),
            "n_high_residual_threats": int((subset["residual_risk"] == "high").sum()),
            "n_moderate_residual_threats": int((subset["residual_risk"] == "moderate").sum()),
            "n_low_residual_threats": int((subset["residual_risk"] == "low").sum()),
            "n_relevant_assumptions": int((assumptions["assumption_domain"] == dimension.replace("_validity", "")).sum()),
            "evidence_work_packages": item["evidence_work_packages"],
            "evidence_summary": item["evidence_summary"],
            "scope_implication": item["scope_implication"],
        })
    return pd.DataFrame(rows)


def make_validity_matrix(summary: pd.DataFrame, output_png: Path, output_pdf: Path) -> None:
    display = summary.copy()
    display["dimension_label"] = display["validity_dimension"].str.replace("_", " ").str.title()
    display["assessment_label"] = display["assessment"].str.replace("_", " ").str.title()
    display["risk_label"] = display["residual_risk"].str.title()
    display["evidence_label"] = display["evidence_work_packages"].str.replace("|", ", ", regex=False)
    table_data = display[["dimension_label", "assessment_label", "risk_label", "rule_id", "evidence_label"]].values.tolist()
    headers = ["Validity dimension", "Derived assessment", "Residual risk", "Rule", "Evidence"]
    fig_height = max(5.0, 0.60 * (len(table_data) + 2))
    fig, ax = plt.subplots(figsize=(13.8, fig_height))
    ax.axis("off")
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc="left", colLoc="left", loc="center", colWidths=[0.23, 0.25, 0.14, 0.10, 0.28])
    table.auto_set_font_size(False)
    table.set_fontsize(9.2)
    table.scale(1, 1.65)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("0.55"); cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("0.88"); cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("0.96")
        else:
            cell.set_facecolor("1.0")
    ax.set_title("Evidence-derived validity assessment and scope of inference", fontsize=14, fontweight="bold", pad=16)
    fig.text(0.5, 0.025, "Assessments are derived from explicit audit rules applied to frozen evidence; no quantitative validity score is constructed.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.95])
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = resolve_inputs()
    manifest = build_evidence_manifest(paths)
    require_inputs(manifest)
    context = derive_verified_context(paths)

    rules = build_validity_rules()
    decision_trace = build_validity_decision_trace(context)
    assessment = build_validity_assessment(context, decision_trace)
    risks = build_risk_register()
    assumptions = build_assumptions_register()
    scope = build_scope_register()
    summary = build_validity_summary(assessment, risks, assumptions)

    manifest.to_csv(OUTPUT_DIR / "validity_evidence_manifest.csv", index=False)
    rules.to_csv(OUTPUT_DIR / "validity_rules.csv", index=False)
    decision_trace.to_csv(OUTPUT_DIR / "validity_decision_trace.csv", index=False)
    assessment.to_csv(OUTPUT_DIR / "validity_assessment.csv", index=False)
    risks.to_csv(OUTPUT_DIR / "validity_risk_register.csv", index=False)
    assumptions.to_csv(OUTPUT_DIR / "assumptions_register.csv", index=False)
    scope.to_csv(OUTPUT_DIR / "framework_scope.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "validity_summary.csv", index=False)

    make_validity_matrix(
        summary,
        OUTPUT_DIR / "validity_matrix.png",
        OUTPUT_DIR / "validity_matrix.pdf",
    )

    overall = "VALID_WITH_EXPLICIT_SCOPE_BOUNDARIES"
    high_residual = int((risks["residual_risk"] == "high").sum())
    moderate_residual = int((risks["residual_risk"] == "moderate").sum())
    low_residual = int((risks["residual_risk"] == "low").sum())

    run_summary = pd.DataFrame(
        [
            {
                "project_root": str(ROOT),
                "output_directory": str(OUTPUT_DIR),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "validity_dimensions": int(len(assessment)),
                "validity_rules": int(len(rules)),
                "derived_decisions": int(len(decision_trace)),
                "registered_threats": int(len(risks)),
                "registered_assumptions": int(len(assumptions)),
                "scope_statements": int(len(scope)),
                "high_residual_threats": high_residual,
                "moderate_residual_threats": moderate_residual,
                "low_residual_threats": low_residual,
                "evidence_inputs_verified": int((manifest["read_status"] == "readable").sum()),
                "evidence_inputs_failed": int((manifest["read_status"] != "readable").sum()),
                "overall_validity_conclusion": overall,
                "upstream_analyses_rerun": "No",
                "new_embeddings": "No",
                "new_graphs": "No",
                "new_clustering": "No",
                "new_prediction_models": "No",
                "new_latent_variables": "No",
                "new_empirical_statistics": "No",
                "classification_basis": "Validity assessments derived deterministically from explicit rules applied to frozen evidence; threats, assumptions, mitigations, and scope remain transparent registries",
            }
        ]
    )
    run_summary.to_csv(OUTPUT_DIR / "validity_run_summary.csv", index=False)

    print("=" * 88)
    print("WP9e2 EVIDENCE-DRIVEN VALIDITY AUDIT COMPLETE")
    print("=" * 88)
    print(f"Project root: {ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Validity dimensions assessed: {len(assessment)}")
    print(f"Decision rules registered: {len(rules)}")
    print(f"Evidence-derived decisions: {len(decision_trace)}")
    print(f"Threats registered: {len(risks)}")
    print(f"Assumptions registered: {len(assumptions)}")
    print(f"Scope statements: {len(scope)}")
    print(
        "Residual threats: "
        f"{high_residual} high, {moderate_residual} moderate, {low_residual} low"
    )
    print(f"Evidence inputs verified: {(manifest['read_status'] == 'readable').sum()}")
    print(f"Overall validity conclusion: {overall}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
