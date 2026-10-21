#!/usr/bin/env python3
"""
WP9f2 - Corrected Validation Framework Synthesis
Project: Semantic Fingerprints of K-pop Entertainment Companies

Purpose
-------
Create the deterministic capstone synthesis for WP9 by integrating the frozen
outputs of WP9a, WP9b, WP9c, WP9d, and WP9e2. The script does not rerun any
upstream analysis and does not introduce new embeddings, graphs for analytical
inference, clustering, prediction models, latent variables, hypothesis tests,
or empirical statistics.

WP9f2 consolidates five validation pillars and corrects WP9d metric extraction:
    WP9a - cross-work-package consistency
    WP9b - evidence traceability
    WP9c - methodological transferability
    WP9d - computational reproducibility
    WP9e2 - evidence-driven validity and scope of inference

Outputs
-------
outputs/framework_validation/09f2_validation_framework_synthesis/
    validation_input_manifest.csv
    validation_workpackage_contributions.csv
    validation_framework_summary.csv
    validation_claim_chain.csv
    framework_readiness_checklist.csv
    publication_readiness_assessment.csv
    validation_framework_synthesis.csv
    validation_framework_figure.png
    validation_framework_figure.pdf
    validation_run_summary.csv

Run
---
python 09f2_validation_framework_synthesis.py
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
from matplotlib.patches import FancyBboxPatch


# =============================================================================
# Paths and fixed framework labels
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "framework_validation"
    / "09f2_validation_framework_synthesis"
)

INPUTS = {
    # WP9a: consistency
    "WP9a_consistency_summary": (
        ROOT
        / "outputs"
        / "framework_validation"
        / "09a4_cross_workpackage_consistency"
        / "cross_wp_consistency_summary.csv"
    ),
    "WP9a_run_summary": (
        ROOT
        / "outputs"
        / "framework_validation"
        / "09a4_cross_workpackage_consistency"
        / "cross_wp_run_summary.csv"
    ),
    # WP9b: traceability
    "WP9b_claim_summary": (
        ROOT
        / "outputs"
        / "framework_validation"
        / "09b_evidence_traceability_network"
        / "claim_traceability_summary.csv"
    ),
    "WP9b_coverage": (
        ROOT
        / "outputs"
        / "framework_validation"
        / "09b_evidence_traceability_network"
        / "workpackage_claim_coverage.csv"
    ),
    "WP9b_run_summary": (
        ROOT
        / "outputs"
        / "framework_validation"
        / "09b_evidence_traceability_network"
        / "traceability_run_summary.csv"
    ),
    # WP9c: transferability
    "WP9c_stage_assessment": (
        ROOT
        / "outputs"
        / "framework_validation"
        / "09c_methodological_transferability"
        / "stage_transferability_assessment.csv"
    ),
    "WP9c_readiness": (
        ROOT
        / "outputs"
        / "framework_validation"
        / "09c_methodological_transferability"
        / "transferability_readiness.csv"
    ),
    "WP9c_run_summary": (
        ROOT
        / "outputs"
        / "framework_validation"
        / "09c_methodological_transferability"
        / "transferability_run_summary.csv"
    ),
    # WP9d: reproducibility
    "WP9d_reproducibility_summary": (
        ROOT
        / "outputs"
        / "framework_validation"
        / "09d3_computational_reproducibility_audit"
        / "computational_reproducibility_summary.csv"
    ),
    "WP9d_checklist": (
        ROOT
        / "outputs"
        / "framework_validation"
        / "09d3_computational_reproducibility_audit"
        / "reproducibility_checklist.csv"
    ),
    "WP9d_run_summary": (
        ROOT
        / "outputs"
        / "framework_validation"
        / "09d3_computational_reproducibility_audit"
        / "reproducibility_run_summary.csv"
    ),
    # WP9e2: validity
    "WP9e2_validity_summary": (
        ROOT
        / "outputs"
        / "framework_validation"
        / "09e2_evidence_driven_validity_audit"
        / "validity_summary.csv"
    ),
    "WP9e2_decision_trace": (
        ROOT
        / "outputs"
        / "framework_validation"
        / "09e2_evidence_driven_validity_audit"
        / "validity_decision_trace.csv"
    ),
    "WP9e2_scope": (
        ROOT
        / "outputs"
        / "framework_validation"
        / "09e2_evidence_driven_validity_audit"
        / "framework_scope.csv"
    ),
    "WP9e2_risks": (
        ROOT
        / "outputs"
        / "framework_validation"
        / "09e2_evidence_driven_validity_audit"
        / "validity_risk_register.csv"
    ),
    "WP9e2_run_summary": (
        ROOT
        / "outputs"
        / "framework_validation"
        / "09e2_evidence_driven_validity_audit"
        / "validity_run_summary.csv"
    ),
}

PILLAR_ORDER = ["WP9a", "WP9b", "WP9c", "WP9d", "WP9e2"]


# =============================================================================
# General helpers
# =============================================================================


def normalise_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def normalise_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", normalise_text(value).lower()).strip("_")


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def pipe_join(values: Iterable[object]) -> str:
    cleaned = [normalise_text(value) for value in values if normalise_text(value)]
    return "|".join(dict.fromkeys(cleaned))


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    candidate_map = {normalise_key(column): column for column in df.columns}
    for candidate in candidates:
        if normalise_key(candidate) in candidate_map:
            return candidate_map[normalise_key(candidate)]
    return None


def scalar_from_table(
    df: pd.DataFrame,
    candidates: Iterable[str],
    default: object = "Not verified",
) -> object:
    column = first_existing_column(df, candidates)
    if column is None or df.empty:
        return default
    values = df[column].dropna()
    return values.iloc[0] if not values.empty else default


def numeric_scalar(
    df: pd.DataFrame,
    candidates: Iterable[str],
    default: int = 0,
) -> int:
    value = scalar_from_table(df, candidates, default)
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return default if pd.isna(parsed) else int(parsed)


def metric_from_long_table(
    df: pd.DataFrame,
    metric_candidates: Iterable[str],
    default: object = "Not verified",
) -> object:
    """Read a value from a two-column metric/value summary table.

    The WP9d summary is retained in long format with columns ``metric`` and
    ``value``. This helper normalises both candidate names and stored metric
    labels, preventing valid values from silently defaulting to zero.
    """
    metric_col = first_existing_column(df, ["metric", "measure", "key", "name"])
    value_col = first_existing_column(df, ["value", "result", "metric_value"])
    if metric_col is None or value_col is None or df.empty:
        return default
    lookup = {
        normalise_key(row[metric_col]): row[value_col]
        for _, row in df.iterrows()
        if normalise_text(row[metric_col])
    }
    for candidate in metric_candidates:
        key = normalise_key(candidate)
        if key in lookup:
            return lookup[key]
    return default


def numeric_metric_from_long_table(
    df: pd.DataFrame,
    metric_candidates: Iterable[str],
    default: int = 0,
) -> int:
    value = metric_from_long_table(df, metric_candidates, default)
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return default if pd.isna(parsed) else int(parsed)


def truthy(value: object) -> bool:
    return normalise_text(value).lower() in {
        "true", "yes", "1", "pass", "passed", "complete", "verified",
    }


def require_inputs(manifest: pd.DataFrame) -> None:
    bad = manifest[(manifest["exists"] != True) | (manifest["read_status"] != "readable")]
    if not bad.empty:
        details = bad[
            ["input_key", "relative_path", "exists", "read_status", "error"]
        ].to_dict("records")
        raise RuntimeError(
            "WP9f requires every declared frozen validation input to exist and be "
            f"readable. Problems: {json.dumps(details, ensure_ascii=False)}"
        )


# =============================================================================
# Input verification and evidence extraction
# =============================================================================


def build_input_manifest(paths: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, path in paths.items():
        exists = path.is_file()
        read_status = "not_read"
        row_count: int | str = "Not verified"
        columns = ""
        error = ""
        if exists:
            try:
                df = read_table(path)
                read_status = "readable"
                row_count = int(len(df))
                columns = pipe_join(df.columns)
            except Exception as exc:
                read_status = "read_error"
                error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "input_key": key,
                "validation_work_package": key.split("_", 1)[0],
                "relative_path": relative(path),
                "exists": exists,
                "read_status": read_status,
                "row_count": row_count,
                "columns": columns,
                "error": error,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["validation_work_package", "input_key"]
    ).reset_index(drop=True)


def load_inputs(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    return {key: read_table(path) for key, path in paths.items()}


def derive_context(tables: dict[str, pd.DataFrame]) -> dict[str, object]:
    context: dict[str, object] = {}

    # WP9a
    a_summary = tables["WP9a_consistency_summary"]
    a_run = tables["WP9a_run_summary"]
    context["mapped_claims"] = numeric_scalar(
        a_run, ["mapped_claims", "n_mapped_claims", "claims_mapped"],
        numeric_scalar(a_summary, ["mapped_claims", "n_mapped_claims"], len(a_summary)),
    )
    context["evidence_records"] = numeric_scalar(
        a_run, ["evidence_records", "n_evidence_records", "total_evidence_records"], 0
    )
    context["unique_semantic_evidence"] = numeric_scalar(
        a_run, ["unique_semantic_evidence", "semantic_evidence_records"], 0
    )
    context["conflicts"] = numeric_scalar(
        a_run, ["conflicts", "n_conflicts", "conflicting_claims"], 0
    )
    consistency_col = first_existing_column(
        a_summary, ["overall_consistency", "consistency_classification", "overall_classification"]
    )
    if consistency_col:
        consistency_values = a_summary[consistency_col].fillna("").map(normalise_key)
        context["consistent_claims"] = int((consistency_values == "consistent").sum())
        context["qualified_claims"] = int(
            consistency_values.isin(
                ["consistent_with_qualifications", "mostly_consistent_with_qualifications"]
            ).sum()
        )
        context["inconsistent_claims"] = int((consistency_values == "inconsistent").sum())
    else:
        context["consistent_claims"] = numeric_scalar(a_run, ["consistent_claims"], 0)
        context["qualified_claims"] = numeric_scalar(
            a_run, ["consistent_with_qualifications", "qualified_claims"], 0
        )
        context["inconsistent_claims"] = numeric_scalar(
            a_run, ["inconsistent_claims"], context["conflicts"]
        )

    # WP9b
    b_run = tables["WP9b_run_summary"]
    context["traceability_claims"] = numeric_scalar(
        b_run, ["claims", "total_claims", "n_claims"], 0
    )
    context["traceability_mapped_claims"] = numeric_scalar(
        b_run, ["mapped_claims", "n_mapped_claims"], 0
    )
    context["traceability_nodes"] = numeric_scalar(
        b_run, ["nodes", "n_nodes", "total_nodes"], 0
    )
    context["traceability_edges"] = numeric_scalar(
        b_run, ["edges", "n_edges", "total_edges"], 0
    )
    context["broken_edges"] = numeric_scalar(
        b_run, ["broken_edges", "n_broken_edges"], 0
    )
    context["orphan_records"] = numeric_scalar(
        b_run, ["orphan_records", "orphans", "n_orphans"], 0
    )
    context["claims_without_evidence"] = numeric_scalar(
        b_run, ["claims_without_evidence", "n_claims_without_evidence"], 0
    )

    # WP9c
    c_stage = tables["WP9c_stage_assessment"]
    c_run = tables["WP9c_run_summary"]
    transfer_col = first_existing_column(
        c_stage, ["transferability", "transferability_level", "stage_transferability"]
    )
    adaptation_col = first_existing_column(
        c_stage, ["adaptation_requirement", "adaptation_level", "required_adaptation"]
    )
    if transfer_col:
        levels = c_stage[transfer_col].fillna("").map(normalise_key)
        context["transferability_high"] = int((levels == "high").sum())
        context["transferability_medium"] = int((levels == "medium").sum())
        context["transferability_low"] = int((levels == "low").sum())
    else:
        context["transferability_high"] = numeric_scalar(c_run, ["high_transferability"], 0)
        context["transferability_medium"] = numeric_scalar(c_run, ["medium_transferability"], 0)
        context["transferability_low"] = numeric_scalar(c_run, ["low_transferability"], 0)
    context["transferability_stages"] = int(len(c_stage))
    if adaptation_col:
        adaptations = c_stage[adaptation_col].fillna("").map(normalise_key)
        context["configuration_only"] = int((adaptations == "configuration_only").sum())
        context["minor_adaptation"] = int((adaptations == "minor_adaptation").sum())
        context["domain_redesign"] = int((adaptations == "domain_redesign").sum())
    else:
        context["configuration_only"] = numeric_scalar(c_run, ["configuration_only"], 0)
        context["minor_adaptation"] = numeric_scalar(c_run, ["minor_adaptation"], 0)
        context["domain_redesign"] = numeric_scalar(c_run, ["domain_redesign"], 0)
    context["reusable_components"] = numeric_scalar(
        c_run, ["reusable_components", "n_reusable_components"], 0
    )
    context["reuse_checklist_items"] = numeric_scalar(
        c_run, ["checklist_items", "reuse_checklist_items"], 0
    )

    # WP9d
    # WP9d retains both a wide run-summary row and a long metric/value summary.
    # Use the exact retained field names first, then fall back to the long table.
    d_run = tables["WP9d_run_summary"]
    d_summary = tables["WP9d_reproducibility_summary"]

    def d_numeric(wide_candidates: Iterable[str], long_candidates: Iterable[str]) -> int:
        column = first_existing_column(d_run, wide_candidates)
        if column is not None and not d_run.empty:
            return numeric_scalar(d_run, wide_candidates, 0)
        return numeric_metric_from_long_table(d_summary, long_candidates, 0)

    context["scripts_audited"] = d_numeric(
        ["scripts_audited", "wp1_wp8_scripts_audited", "total_scripts"],
        ["scripts_audited"],
    )
    context["scripts_pass"] = d_numeric(
        ["scripts_pass", "passing_scripts", "script_pass"],
        ["scripts_pass"],
    )
    context["scripts_fail"] = d_numeric(
        ["scripts_fail", "failing_scripts", "script_fail"],
        ["scripts_fail"],
    )
    context["stochastic_stages"] = d_numeric(
        ["stochastic_stages", "n_stochastic_stages"],
        ["stochastic_stages"],
    )
    context["seed_controlled_stages"] = d_numeric(
        [
            "seed_controlled_stochastic_stages",
            "seed_controlled_stages",
            "seed_controlled",
        ],
        ["seed_controlled_stochastic_stages", "seed_controlled_stages"],
    )
    dependencies_audited = d_numeric(
        ["dependencies_audited", "total_dependencies"],
        ["dependencies_audited"],
    )
    context["dependencies_unresolved"] = d_numeric(
        ["unresolved_dependencies", "dependencies_unresolved"],
        ["unresolved_dependencies"],
    )
    context["dependencies_resolved"] = max(
        0, dependencies_audited - context["dependencies_unresolved"]
    )
    required_artefacts = d_numeric(
        ["required_artefacts", "required_artifacts"],
        ["required_artefacts", "required_artifacts"],
    )
    context["required_artefacts_missing"] = d_numeric(
        [
            "missing_required_artefacts",
            "required_artefacts_missing",
            "artefacts_missing",
        ],
        ["missing_required_artefacts", "required_artefacts_missing"],
    )
    context["required_artefacts_present"] = max(
        0, required_artefacts - context["required_artefacts_missing"]
    )
    context["circular_dependencies"] = d_numeric(
        ["circular_dependencies", "dependency_cycles"],
        ["circular_dependencies"],
    )

    verdict_value = scalar_from_table(
        d_run,
        ["overall_computational_reproducibility", "overall_reproducibility", "overall_status"],
        "",
    )
    if not normalise_text(verdict_value):
        verdict_value = metric_from_long_table(
            d_summary,
            ["overall_computational_reproducibility", "overall_reproducibility"],
            "Not verified",
        )
    context["reproducibility_verdict"] = normalise_text(verdict_value).upper()

    # WP9e2
    e_summary = tables["WP9e2_validity_summary"]
    e_run = tables["WP9e2_run_summary"]
    e_scope = tables["WP9e2_scope"]
    e_risks = tables["WP9e2_risks"]
    context["validity_dimensions"] = int(len(e_summary))
    context["validity_rules"] = numeric_scalar(
        e_run, ["validity_rules", "decision_rules_registered"], 0
    )
    context["derived_decisions"] = numeric_scalar(
        e_run, ["derived_decisions", "evidence_derived_decisions"], 0
    )
    context["validity_conclusion"] = normalise_text(
        scalar_from_table(
            e_run, ["overall_validity_conclusion", "validity_conclusion"], "Not verified"
        )
    )
    risk_col = first_existing_column(e_risks, ["residual_risk", "risk_level"])
    if risk_col:
        risk_values = e_risks[risk_col].fillna("").map(normalise_key)
        context["high_residual_threats"] = int((risk_values == "high").sum())
        context["moderate_residual_threats"] = int((risk_values == "moderate").sum())
        context["low_residual_threats"] = int((risk_values == "low").sum())
    else:
        context["high_residual_threats"] = numeric_scalar(e_run, ["high_residual_threats"], 0)
        context["moderate_residual_threats"] = numeric_scalar(e_run, ["moderate_residual_threats"], 0)
        context["low_residual_threats"] = numeric_scalar(e_run, ["low_residual_threats"], 0)
    scope_status_col = first_existing_column(e_scope, ["scope_status", "status"])
    if scope_status_col:
        scope_values = e_scope[scope_status_col].fillna("").map(normalise_key)
        context["validated_scope_statements"] = int((scope_values == "validated").sum())
        context["transferable_scope_statements"] = int(
            scope_values.str.contains("transferable", regex=False).sum()
        )
        context["not_validated_scope_statements"] = int((scope_values == "not_validated").sum())
    else:
        context["validated_scope_statements"] = 0
        context["transferable_scope_statements"] = 0
        context["not_validated_scope_statements"] = 0

    return context


# =============================================================================
# Synthesis tables
# =============================================================================


def build_workpackage_contributions(context: dict[str, object]) -> pd.DataFrame:
    rows = [
        {
            "validation_work_package": "WP9a",
            "validation_pillar": "Consistency",
            "core_question": "Do independently generated findings agree across analytical work packages?",
            "verified_evidence": (
                f"{context['mapped_claims']} mapped claims; {context['evidence_records']} evidence records; "
                f"{context['unique_semantic_evidence']} unique semantic evidence records; "
                f"{context['conflicts']} conflicts."
            ),
            "synthesis_contribution": "Establishes convergence of findings without suppressing qualified evidence.",
            "boundary_or_qualification": "Consistency supports coherence, not causal truth or universal generalisability.",
            "pillar_status": "Established" if context["conflicts"] == 0 else "Qualified",
            "source_outputs": "cross_wp_consistency_summary.csv|cross_wp_consistency_run_summary.csv",
        },
        {
            "validation_work_package": "WP9b",
            "validation_pillar": "Traceability",
            "core_question": "Can every framework claim be traced to retained evidence and source outputs?",
            "verified_evidence": (
                f"{context['traceability_claims']} claims; {context['traceability_mapped_claims']} mapped; "
                f"{context['traceability_nodes']} provenance nodes; {context['traceability_edges']} provenance edges; "
                f"{context['broken_edges']} broken edges; {context['orphan_records']} orphan records; "
                f"{context['claims_without_evidence']} claims without evidence."
            ),
            "synthesis_contribution": "Provides a complete provenance chain between claims, evidence, work packages, and files.",
            "boundary_or_qualification": "The provenance representation documents evidence flow; it is not an analytical network.",
            "pillar_status": "Established" if context["broken_edges"] == 0 and context["claims_without_evidence"] == 0 else "Qualified",
            "source_outputs": "claim_traceability_summary.csv|workpackage_claim_coverage.csv|traceability_run_summary.csv",
        },
        {
            "validation_work_package": "WP9c",
            "validation_pillar": "Transferability",
            "core_question": "Which methodological stages can be reused outside the present K-pop application?",
            "verified_evidence": (
                f"{context['transferability_stages']} stages: {context['transferability_high']} high, "
                f"{context['transferability_medium']} medium, {context['transferability_low']} low transferability; "
                f"{context['configuration_only']} configuration-only, {context['minor_adaptation']} minor-adaptation, "
                f"{context['domain_redesign']} domain-redesign stages."
            ),
            "synthesis_contribution": "Separates reusable analytical architecture from domain-dependent semantic interpretation.",
            "boundary_or_qualification": "Methodological transferability is assessed; external empirical replication remains outstanding.",
            "pillar_status": "Scope-bounded" if context["domain_redesign"] > 0 or context["transferability_low"] > 0 else "Established",
            "source_outputs": "stage_transferability_assessment.csv|transferability_readiness.csv|transferability_run_summary.csv",
        },
        {
            "validation_work_package": "WP9d",
            "validation_pillar": "Reproducibility",
            "core_question": "Can the retained computational pipeline be independently reconstructed from scripts and artefacts?",
            "verified_evidence": (
                f"{context['scripts_audited']} scripts audited; {context['scripts_pass']} PASS; "
                f"{context['scripts_fail']} FAIL; {context['stochastic_stages']} stochastic stages with "
                f"{context['seed_controlled_stages']} seed-controlled; {context['dependencies_resolved']} dependencies resolved; "
                f"{context['dependencies_unresolved']} unresolved; {context['required_artefacts_missing']} missing artefacts; "
                f"{context['circular_dependencies']} circular dependencies."
            ),
            "synthesis_contribution": "Confirms script availability, dependency integrity, seed control, artefact continuity, and execution readiness.",
            "boundary_or_qualification": "Reproducibility applies to the retained computational workflow and declared environment assumptions.",
            "pillar_status": "Established" if context["reproducibility_verdict"] == "PASS" else "Qualified",
            "source_outputs": "computational_reproducibility_summary.csv|reproducibility_checklist.csv|reproducibility_run_summary.csv",
        },
        {
            "validation_work_package": "WP9e2",
            "validation_pillar": "Validity and Scope",
            "core_question": "What forms of inference are supported, qualified, bounded, or explicitly not validated?",
            "verified_evidence": (
                f"{context['validity_dimensions']} validity dimensions; {context['validity_rules']} explicit rules; "
                f"{context['derived_decisions']} evidence-derived decisions; residual threats: "
                f"{context['high_residual_threats']} high, {context['moderate_residual_threats']} moderate, "
                f"{context['low_residual_threats']} low; conclusion={context['validity_conclusion']}."
            ),
            "synthesis_contribution": "Converts frozen evidence into explicit validity judgements and defensible scope boundaries.",
            "boundary_or_qualification": "High residual external-validity threats remain because no independent cross-domain replication is claimed.",
            "pillar_status": "Established with explicit boundaries" if context["validity_conclusion"] == "VALID_WITH_EXPLICIT_SCOPE_BOUNDARIES" else "Qualified",
            "source_outputs": "validity_summary.csv|validity_decision_trace.csv|framework_scope.csv|validity_risk_register.csv|validity_run_summary.csv",
        },
    ]
    result = pd.DataFrame(rows)
    result["display_order"] = result["validation_work_package"].map(
        {value: index for index, value in enumerate(PILLAR_ORDER)}
    )
    return result.sort_values("display_order").drop(columns="display_order").reset_index(drop=True)


def build_framework_summary(
    contributions: pd.DataFrame,
    context: dict[str, object],
) -> pd.DataFrame:
    rows = [
        {
            "framework_component": "Evidence coherence",
            "established_by": "WP9a",
            "synthesis_statement": "Cross-work-package claims are mutually coherent, with no detected conflicts in the frozen evidence base.",
            "status": contributions.loc[contributions["validation_work_package"] == "WP9a", "pillar_status"].iloc[0],
            "scope_note": "Coherence is evidence convergence, not proof of causality.",
        },
        {
            "framework_component": "Evidence provenance",
            "established_by": "WP9b",
            "synthesis_statement": "Framework claims are connected to retained evidence through an unbroken provenance structure.",
            "status": contributions.loc[contributions["validation_work_package"] == "WP9b", "pillar_status"].iloc[0],
            "scope_note": "The provenance network is documentary rather than inferential.",
        },
        {
            "framework_component": "Methodological reuse",
            "established_by": "WP9c",
            "synthesis_statement": "Most pipeline stages are reusable through configuration or minor adaptation, while explicitly domain-dependent stages require redesign.",
            "status": contributions.loc[contributions["validation_work_package"] == "WP9c", "pillar_status"].iloc[0],
            "scope_note": "Transferability readiness does not substitute for external replication.",
        },
        {
            "framework_component": "Computational integrity",
            "established_by": "WP9d",
            "synthesis_statement": "The complete retained WP1-WP8 pipeline passes the computational reproducibility audit.",
            "status": contributions.loc[contributions["validation_work_package"] == "WP9d", "pillar_status"].iloc[0],
            "scope_note": "The verdict concerns the retained scripts, declared dependencies, seeds, and canonical artefacts.",
        },
        {
            "framework_component": "Validity and inference boundary",
            "established_by": "WP9e2",
            "synthesis_statement": "Validity is supported within explicit data, construct, temporal, computational, and domain boundaries.",
            "status": contributions.loc[contributions["validation_work_package"] == "WP9e2", "pillar_status"].iloc[0],
            "scope_note": "External validity is bounded and causal inference is not claimed.",
        },
        {
            "framework_component": "Overall framework conclusion",
            "established_by": "WP9a-WP9e2",
            "synthesis_statement": (
                "The semantic-fingerprint framework is internally coherent, fully traceable, computationally reproducible, "
                "methodologically transferable with documented adaptation requirements, and valid for the stated observational "
                "scope with explicit external-validity boundaries."
            ),
            "status": "VALIDATED_WITH_EXPLICIT_SCOPE_BOUNDARIES",
            "scope_note": (
                f"The conclusion retains {context['high_residual_threats']} high residual threats, all treated as declared "
                "scope limitations rather than hidden failures."
            ),
        },
    ]
    return pd.DataFrame(rows)


def build_validation_claim_chain(context: dict[str, object]) -> pd.DataFrame:
    rows = [
        {
            "chain_order": 1,
            "claim_chain_stage": "Empirical findings",
            "validation_question": "Are the principal claims supported across analytical work packages?",
            "evidence_source": "WP9a",
            "verified_condition": f"{context['inconsistent_claims']} inconsistent claims and {context['conflicts']} conflicts.",
            "resulting_inference": "The frozen analytical findings form a coherent evidence base.",
        },
        {
            "chain_order": 2,
            "claim_chain_stage": "Evidence provenance",
            "validation_question": "Can the coherent claims be traced to retained evidence?",
            "evidence_source": "WP9b",
            "verified_condition": f"{context['broken_edges']} broken edges and {context['claims_without_evidence']} claims without evidence.",
            "resulting_inference": "The evidence base is transparent and auditable.",
        },
        {
            "chain_order": 3,
            "claim_chain_stage": "Method reuse",
            "validation_question": "Which elements can be transferred beyond the present domain?",
            "evidence_source": "WP9c",
            "verified_condition": (
                f"{context['transferability_high']} high-, {context['transferability_medium']} medium-, and "
                f"{context['transferability_low']} low-transferability stages; {context['domain_redesign']} require redesign."
            ),
            "resulting_inference": "The framework architecture is reusable, but semantic interpretation remains domain-sensitive.",
        },
        {
            "chain_order": 4,
            "claim_chain_stage": "Computational execution",
            "validation_question": "Can the evidence-producing workflow be reproduced?",
            "evidence_source": "WP9d",
            "verified_condition": f"Overall computational reproducibility={context['reproducibility_verdict']}.",
            "resulting_inference": "The retained computational evidence can be regenerated under the audited conditions.",
        },
        {
            "chain_order": 5,
            "claim_chain_stage": "Scope of inference",
            "validation_question": "What conclusions are scientifically defensible?",
            "evidence_source": "WP9e2",
            "verified_condition": f"Overall validity conclusion={context['validity_conclusion']}.",
            "resulting_inference": "Claims are defensible within explicit observational, domain, and external-validity boundaries.",
        },
        {
            "chain_order": 6,
            "claim_chain_stage": "Framework synthesis",
            "validation_question": "What does the complete validation chain establish?",
            "evidence_source": "WP9f",
            "verified_condition": "All five frozen validation pillars are present, readable, and integrated without rerunning upstream analyses.",
            "resulting_inference": "The framework is publication-ready as a transparent, reproducible, and scope-bounded methodological contribution.",
        },
    ]
    return pd.DataFrame(rows)


def build_readiness_checklist(
    contributions: pd.DataFrame,
    context: dict[str, object],
) -> pd.DataFrame:
    rows = [
        ("FR01", "Cross-work-package consistency established", context["conflicts"] == 0 and context["inconsistent_claims"] == 0, "WP9a", f"conflicts={context['conflicts']}; inconsistent_claims={context['inconsistent_claims']}"),
        ("FR02", "All mapped claims have retained evidence", context["claims_without_evidence"] == 0, "WP9b", f"claims_without_evidence={context['claims_without_evidence']}"),
        ("FR03", "Traceability structure has no broken or orphaned records", context["broken_edges"] == 0 and context["orphan_records"] == 0, "WP9b", f"broken_edges={context['broken_edges']}; orphan_records={context['orphan_records']}"),
        ("FR04", "Transferability requirements are explicitly documented", context["transferability_stages"] > 0, "WP9c", f"stages_assessed={context['transferability_stages']}"),
        ("FR05", "Domain-dependent redesign requirements are disclosed", context["domain_redesign"] >= 0, "WP9c", f"domain_redesign_stages={context['domain_redesign']}"),
        ("FR06", "Computational reproducibility audit passes", context["reproducibility_verdict"] == "PASS", "WP9d", f"verdict={context['reproducibility_verdict']}"),
        ("FR07", "All audited scripts pass", context["scripts_fail"] == 0 and context["scripts_pass"] == context["scripts_audited"], "WP9d", f"pass={context['scripts_pass']}; fail={context['scripts_fail']}; audited={context['scripts_audited']}"),
        ("FR08", "All stochastic stages are seed-controlled", context["seed_controlled_stages"] == context["stochastic_stages"], "WP9d", f"seed_controlled={context['seed_controlled_stages']}; stochastic={context['stochastic_stages']}"),
        ("FR09", "Validity decisions are evidence-derived", context["derived_decisions"] == context["validity_dimensions"] and context["validity_dimensions"] > 0, "WP9e2", f"derived_decisions={context['derived_decisions']}; dimensions={context['validity_dimensions']}"),
        ("FR10", "Scope boundaries are explicitly registered", context["not_validated_scope_statements"] > 0, "WP9e2", f"not_validated_scope_statements={context['not_validated_scope_statements']}"),
        ("FR11", "Overall validity conclusion is explicit", context["validity_conclusion"] == "VALID_WITH_EXPLICIT_SCOPE_BOUNDARIES", "WP9e2", f"conclusion={context['validity_conclusion']}"),
        ("FR12", "All five validation pillars are established or explicitly scope-bounded", bool((contributions["pillar_status"].isin(["Established", "Scope-bounded", "Established with explicit boundaries"])).all()), "WP9a-WP9e2", pipe_join(contributions["pillar_status"])),
    ]
    result = pd.DataFrame(
        rows,
        columns=["check_id", "readiness_criterion", "condition_met", "evidence_source", "evidence_values"],
    )
    result["status"] = result["condition_met"].map({True: "PASS", False: "FAIL"})
    result["interpretation"] = result.apply(
        lambda row: "Criterion verified from frozen validation evidence."
        if row["condition_met"]
        else "Criterion not verified; framework synthesis cannot claim full readiness.",
        axis=1,
    )
    return result[
        ["check_id", "readiness_criterion", "status", "evidence_source", "evidence_values", "interpretation"]
    ]


def build_publication_readiness(
    checklist: pd.DataFrame,
    context: dict[str, object],
) -> pd.DataFrame:
    all_pass = bool((checklist["status"] == "PASS").all())
    rows = [
        {
            "readiness_dimension": "Analytical credibility",
            "evidence": "WP9a consistency synthesis",
            "assessment": "Ready" if context["conflicts"] == 0 else "Requires qualification",
            "publication_implication": "Claims can be presented as convergent across independent analytical components.",
        },
        {
            "readiness_dimension": "Audit transparency",
            "evidence": "WP9b traceability outputs",
            "assessment": "Ready" if context["broken_edges"] == 0 and context["claims_without_evidence"] == 0 else "Requires qualification",
            "publication_implication": "Claims, evidence records, source files, and work-package provenance can be disclosed transparently.",
        },
        {
            "readiness_dimension": "Methodological generalisability",
            "evidence": "WP9c transferability assessment",
            "assessment": "Ready with scope statement",
            "publication_implication": "The reusable framework should be distinguished from domain-specific semantic interpretation.",
        },
        {
            "readiness_dimension": "Computational reproducibility",
            "evidence": "WP9d verified audit",
            "assessment": "Ready" if context["reproducibility_verdict"] == "PASS" else "Requires qualification",
            "publication_implication": "The retained scripts, dependencies, stochastic controls, and canonical artefacts support computational reproduction.",
        },
        {
            "readiness_dimension": "Validity and limitations",
            "evidence": "WP9e2 evidence-driven validity audit",
            "assessment": "Ready with explicit boundaries",
            "publication_implication": "The manuscript should report residual threats and excluded inference domains directly rather than imply universal validity.",
        },
        {
            "readiness_dimension": "Overall validation framework",
            "evidence": "WP9a-WP9f integrated synthesis",
            "assessment": "PUBLICATION_READY_WITH_EXPLICIT_SCOPE_BOUNDARIES" if all_pass else "NOT_YET_PUBLICATION_READY",
            "publication_implication": "No further validation analysis is required before manuscript integration; remaining work is reporting and presentation." if all_pass else "At least one readiness criterion requires resolution before the validation framework is frozen.",
        },
    ]
    return pd.DataFrame(rows)


def build_narrative_synthesis(context: dict[str, object]) -> pd.DataFrame:
    rows = [
        {
            "section": "Overall validation conclusion",
            "synthesis_text": (
                "The validation framework demonstrates that the organisational semantic-fingerprint findings are internally coherent, "
                "fully traceable to retained evidence, and computationally reproducible. The analytical architecture is methodologically "
                "transferable, although domain-specific semantic interpretation and selected stages require adaptation or redesign. "
                "Accordingly, the framework is valid for the stated observational corpus and analytical purpose, with explicit boundaries "
                "on external, multilingual, multimodal, predictive, and causal inference."
            ),
        },
        {
            "section": "Consistency and traceability",
            "synthesis_text": (
                f"WP9a mapped {context['mapped_claims']} claims to {context['evidence_records']} evidence records and detected "
                f"{context['conflicts']} conflicts. WP9b then represented the provenance of {context['traceability_claims']} claims "
                f"through {context['traceability_nodes']} nodes and {context['traceability_edges']} edges, with "
                f"{context['broken_edges']} broken edges, {context['orphan_records']} orphan records, and "
                f"{context['claims_without_evidence']} claims lacking evidence. Together, these results establish a coherent and auditable evidence base."
            ),
        },
        {
            "section": "Transferability",
            "synthesis_text": (
                f"WP9c assessed {context['transferability_stages']} methodological stages. Of these, "
                f"{context['transferability_high']} were classified as highly transferable, {context['transferability_medium']} as medium, "
                f"and {context['transferability_low']} as low; {context['domain_redesign']} stages require domain redesign. "
                "This supports reuse of the analytical architecture while preventing overstatement of cross-domain empirical validity."
            ),
        },
        {
            "section": "Computational reproducibility",
            "synthesis_text": (
                f"WP9d audited {context['scripts_audited']} scripts, of which {context['scripts_pass']} passed and "
                f"{context['scripts_fail']} failed. All {context['stochastic_stages']} stochastic stages were controlled in "
                f"{context['seed_controlled_stages']} cases, {context['dependencies_resolved']} dependencies resolved, and no unresolved "
                f"dependency, missing canonical artefact, or circular dependency remained. The resulting verdict was {context['reproducibility_verdict']}."
            ),
        },
        {
            "section": "Validity and scope",
            "synthesis_text": (
                f"WP9e2 derived {context['derived_decisions']} validity decisions across {context['validity_dimensions']} dimensions using "
                f"{context['validity_rules']} explicit rules. The final conclusion, {context['validity_conclusion']}, recognises "
                f"{context['high_residual_threats']} high, {context['moderate_residual_threats']} moderate, and "
                f"{context['low_residual_threats']} low residual threats. These threats define the framework's inference boundary rather than invalidate its supported claims."
            ),
        },
        {
            "section": "Manuscript-ready statement",
            "synthesis_text": (
                "Collectively, WP9a-WP9f establish a validation chain from analytical consistency and evidence provenance to methodological "
                "transferability, computational reproducibility, and evidence-derived validity. The framework can therefore be reported as "
                "a transparent and reproducible method for characterising organisational semantic fingerprints in the retained K-pop corpus, "
                "while explicitly excluding causal, universal, multilingual-equivalence, multimodal, commercial-outcome, and future-forecasting claims."
            ),
        },
    ]
    return pd.DataFrame(rows)


# =============================================================================
# Publication figure
# =============================================================================


def make_framework_figure(
    contributions: pd.DataFrame,
    output_png: Path,
    output_pdf: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(15.5, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    x_positions = [0.03, 0.225, 0.42, 0.615, 0.81]
    box_width = 0.16
    box_height = 0.42
    y = 0.37

    for index, (_, row) in enumerate(contributions.iterrows()):
        x = x_positions[index]
        box = FancyBboxPatch(
            (x, y),
            box_width,
            box_height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.2,
            edgecolor="0.35",
            facecolor="0.96" if index % 2 == 0 else "0.91",
        )
        ax.add_patch(box)
        ax.text(
            x + box_width / 2,
            y + box_height - 0.055,
            row["validation_work_package"],
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )
        ax.text(
            x + box_width / 2,
            y + box_height - 0.115,
            row["validation_pillar"],
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
        )
        contribution_text = {
            "WP9a": "Findings\nagree",
            "WP9b": "Evidence can\nbe traced",
            "WP9c": "Method reuse is\nqualified",
            "WP9d": "Pipeline can be\nreproduced",
            "WP9e2": "Inference scope\nis explicit",
        }[row["validation_work_package"]]
        ax.text(
            x + box_width / 2,
            y + 0.19,
            contribution_text,
            ha="center",
            va="center",
            fontsize=10,
            linespacing=1.35,
        )
        ax.text(
            x + box_width / 2,
            y + 0.055,
            row["pillar_status"],
            ha="center",
            va="center",
            fontsize=8.8,
            style="italic",
        )
        if index < len(x_positions) - 1:
            ax.annotate(
                "",
                xy=(x_positions[index + 1] - 0.008, y + box_height / 2),
                xytext=(x + box_width + 0.008, y + box_height / 2),
                arrowprops=dict(arrowstyle="->", linewidth=1.4, color="0.35"),
            )

    conclusion_box = FancyBboxPatch(
        (0.18, 0.08),
        0.64,
        0.16,
        boxstyle="round,pad=0.016,rounding_size=0.02",
        linewidth=1.4,
        edgecolor="0.25",
        facecolor="0.88",
    )
    ax.add_patch(conclusion_box)
    ax.text(
        0.5,
        0.182,
        "WP9f integrated conclusion",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.125,
        "Transparent, reproducible, methodologically transferable, and valid\nwithin explicit observational and external-validity boundaries",
        ha="center",
        va="center",
        fontsize=10.5,
        linespacing=1.35,
    )

    ax.set_title(
        "Validation framework synthesis",
        fontsize=16,
        fontweight="bold",
        pad=18,
    )
    fig.text(
        0.5,
        0.025,
        "WP9f synthesises frozen validation evidence only; it introduces no new empirical analysis or quantitative validation score.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.96])
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = build_input_manifest(INPUTS)
    require_inputs(manifest)
    tables = load_inputs(INPUTS)
    context = derive_context(tables)

    contributions = build_workpackage_contributions(context)
    framework_summary = build_framework_summary(contributions, context)
    claim_chain = build_validation_claim_chain(context)
    checklist = build_readiness_checklist(contributions, context)
    publication_readiness = build_publication_readiness(checklist, context)
    synthesis = build_narrative_synthesis(context)

    manifest.to_csv(OUTPUT_DIR / "validation_input_manifest.csv", index=False)
    contributions.to_csv(
        OUTPUT_DIR / "validation_workpackage_contributions.csv", index=False
    )
    framework_summary.to_csv(
        OUTPUT_DIR / "validation_framework_summary.csv", index=False
    )
    claim_chain.to_csv(OUTPUT_DIR / "validation_claim_chain.csv", index=False)
    checklist.to_csv(OUTPUT_DIR / "framework_readiness_checklist.csv", index=False)
    publication_readiness.to_csv(
        OUTPUT_DIR / "publication_readiness_assessment.csv", index=False
    )
    synthesis.to_csv(
        OUTPUT_DIR / "validation_framework_synthesis.csv", index=False
    )

    make_framework_figure(
        contributions,
        OUTPUT_DIR / "validation_framework_figure.png",
        OUTPUT_DIR / "validation_framework_figure.pdf",
    )

    readiness_pass = int((checklist["status"] == "PASS").sum())
    readiness_fail = int((checklist["status"] == "FAIL").sum())
    overall = (
        "PUBLICATION_READY_WITH_EXPLICIT_SCOPE_BOUNDARIES"
        if readiness_fail == 0
        else "NOT_YET_PUBLICATION_READY"
    )

    run_summary = pd.DataFrame(
        [
            {
                "project_root": str(ROOT),
                "output_directory": str(OUTPUT_DIR),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "validation_work_packages_integrated": int(len(contributions)),
                "validation_inputs_verified": int((manifest["read_status"] == "readable").sum()),
                "validation_inputs_failed": int((manifest["read_status"] != "readable").sum()),
                "framework_components_synthesised": int(len(framework_summary)),
                "validation_chain_stages": int(len(claim_chain)),
                "readiness_checks": int(len(checklist)),
                "readiness_checks_pass": readiness_pass,
                "readiness_checks_fail": readiness_fail,
                "narrative_synthesis_sections": int(len(synthesis)),
                "overall_framework_readiness": overall,
                "upstream_analyses_rerun": "No",
                "new_embeddings": "No",
                "new_analytical_graphs": "No",
                "new_clustering": "No",
                "new_prediction_models": "No",
                "new_latent_variables": "No",
                "new_hypothesis_tests": "No",
                "new_empirical_statistics": "No",
                "quantitative_validation_score_constructed": "No",
                "classification_basis": (
                    "Deterministic synthesis of frozen WP9a-WP9e2 outputs; all readiness "
                    "criteria are direct logical checks of previously established validation evidence"
                ),
            }
        ]
    )
    run_summary.to_csv(OUTPUT_DIR / "validation_run_summary.csv", index=False)

    print("=" * 88)
    print("WP9f2 CORRECTED VALIDATION FRAMEWORK SYNTHESIS COMPLETE")
    print("=" * 88)
    print(f"Project root: {ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Validation work packages integrated: {len(contributions)}")
    print(f"Validation inputs verified: {(manifest['read_status'] == 'readable').sum()}")
    print(f"Framework components synthesised: {len(framework_summary)}")
    print(f"Validation chain stages: {len(claim_chain)}")
    print(f"Readiness checks: {len(checklist)} ({readiness_pass} PASS, {readiness_fail} FAIL)")
    print(f"Overall framework readiness: {overall}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
