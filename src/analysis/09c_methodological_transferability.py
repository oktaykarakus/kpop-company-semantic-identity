"""
09c_methodological_transferability.py

WP9c — Methodological Transferability
Project: Semantic Fingerprints of K-pop Entertainment Companies

Scientific question
-------------------
To what extent can the frozen semantic-fingerprinting framework be reused in
another organisational text domain without changing its analytical logic?

Principles
----------
- WP1-WP8, WP9a, and WP9b are treated as frozen.
- No new embeddings, graphs, clustering, prediction models, latent variables,
  hypothesis tests, effect sizes, or composite empirical scores are created.
- Transferability is assessed as a deterministic methodological audit.
- The audit distinguishes domain-specific inputs and interpretation from
  reusable analytical operations.
- Classifications are predefined in this script and are not inferred from the
  K-pop results.
- Application domains are not proposed or ranked.

Canonical inputs
----------------
WP8 (resolved uniquely below outputs/):
    integrated_company_identity.csv
    company_identity_summary.csv

WP9a:
outputs/framework_validation/09a4_cross_workpackage_consistency/
    cross_wp_consistency_summary.csv

WP9b:
outputs/framework_validation/09b_evidence_traceability_network/
    traceability_input_manifest.csv
    claim_traceability_summary.csv
    traceability_run_summary.csv

Outputs
-------
outputs/framework_validation/09c_methodological_transferability/
    methodology_stage_inventory.csv
    stage_transferability_assessment.csv
    framework_transferability_summary.csv
    domain_dependency_matrix.csv
    transferability_readiness.csv
    reusable_pipeline_components.csv
    framework_reuse_checklist.csv
    transferability_input_manifest.csv
    transferability_run_summary.csv

Run
---
python 09c_methodological_transferability.py
"""

from __future__ import annotations

from pathlib import Path
import json
import math
import re
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

WP9B_DIR = (
    ROOT
    / "outputs"
    / "framework_validation"
    / "09b_evidence_traceability_network"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "framework_validation"
    / "09c_methodological_transferability"
)

DIRECT_INPUTS = {
    "WP9a_consistency_summary": WP9A_DIR / "cross_wp_consistency_summary.csv",
    "WP9b_input_manifest": WP9B_DIR / "traceability_input_manifest.csv",
    "WP9b_claim_summary": WP9B_DIR / "claim_traceability_summary.csv",
    "WP9b_run_summary": WP9B_DIR / "traceability_run_summary.csv",
}

WP8_FILENAMES = {
    "WP8_integrated_identity": "integrated_company_identity.csv",
    "WP8_company_summary": "company_identity_summary.csv",
}

TRANSFERABILITY_ORDER = {"high": 0, "medium": 1, "low": 2}
MODIFICATION_ORDER = {
    "none": 0,
    "configuration_only": 1,
    "minor_adaptation": 2,
    "substantive_domain_redesign": 3,
}


# =============================================================================
# General helpers
# =============================================================================


def normalise_text(value: object) -> str:
    """Return a stable whitespace-normalised string."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


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
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


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


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = normalise_text(value).lower()
    return text in {"true", "1", "yes", "y"}


def first_numeric_value(df: pd.DataFrame, candidates: Iterable[str]) -> float:
    for column in candidates:
        if column not in df.columns or df.empty:
            continue
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if not values.empty:
            return float(values.iloc[0])
    return np.nan


# =============================================================================
# Input resolution and validation
# =============================================================================


def resolve_inputs() -> dict[str, Path]:
    paths = dict(DIRECT_INPUTS)
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
                    if key in DIRECT_INPUTS
                    else "unique_exact_filename"
                ),
                "role_in_wp9c": {
                    "WP8_integrated_identity": "verifies frozen identity synthesis output",
                    "WP8_company_summary": "verifies frozen company-level synthesis output",
                    "WP9a_consistency_summary": "verifies frozen cross-work-package validation status",
                    "WP9b_input_manifest": "verifies upstream input provenance",
                    "WP9b_claim_summary": "verifies claim-to-evidence traceability coverage",
                    "WP9b_run_summary": "verifies structural traceability integrity",
                }[key],
            }
        )
    return pd.DataFrame(rows).sort_values(["work_package", "input_key"])


def load_inputs(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    return {
        "wp8_integrated": read_csv_required(paths["WP8_integrated_identity"], []),
        "wp8_summary": read_csv_required(paths["WP8_company_summary"], []),
        "wp9a_summary": read_csv_required(
            paths["WP9a_consistency_summary"],
            ["claim_id", "company", "claim_category", "overall_consistency"],
        ),
        "wp9b_manifest": read_csv_required(
            paths["WP9b_input_manifest"],
            ["input_key", "work_package", "relative_path", "exists"],
        ),
        "wp9b_claims": read_csv_required(
            paths["WP9b_claim_summary"],
            [
                "claim_id",
                "company",
                "is_mapped_claim",
                "has_complete_source_path",
            ],
        ),
        "wp9b_run": read_csv_required(paths["WP9b_run_summary"], []),
    }


def validate_frozen_inputs(data: dict[str, pd.DataFrame]) -> dict[str, object]:
    wp9a = data["wp9a_summary"].copy()
    wp9b_claims = data["wp9b_claims"].copy()
    wp9b_manifest = data["wp9b_manifest"].copy()

    if wp9a["claim_id"].duplicated().any():
        duplicates = wp9a.loc[wp9a["claim_id"].duplicated(), "claim_id"].tolist()
        raise ValueError(f"Duplicate WP9a claim IDs detected: {duplicates}")

    if wp9b_claims["claim_id"].duplicated().any():
        duplicates = wp9b_claims.loc[
            wp9b_claims["claim_id"].duplicated(), "claim_id"
        ].tolist()
        raise ValueError(f"Duplicate WP9b claim IDs detected: {duplicates}")

    missing_wp9a_in_wp9b = sorted(
        set(wp9a["claim_id"].astype(str))
        - set(wp9b_claims["claim_id"].astype(str))
    )
    if missing_wp9a_in_wp9b:
        raise ValueError(
            "Mapped WP9a claims missing from WP9b traceability summary: "
            f"{missing_wp9a_in_wp9b}"
        )

    manifest_missing = wp9b_manifest.loc[~wp9b_manifest["exists"].map(as_bool)]
    if not manifest_missing.empty:
        raise FileNotFoundError(
            "WP9b manifest contains unresolved inputs: "
            + pipe_join(manifest_missing["relative_path"])
        )

    mapped = wp9b_claims["is_mapped_claim"].map(as_bool)
    complete = wp9b_claims["has_complete_source_path"].map(as_bool)

    return {
        "n_wp8_integrated_rows": int(len(data["wp8_integrated"])),
        "n_wp8_summary_rows": int(len(data["wp8_summary"])),
        "n_wp9a_mapped_claims": int(len(wp9a)),
        "n_wp9b_claims": int(len(wp9b_claims)),
        "n_wp9b_mapped_claims": int(mapped.sum()),
        "n_wp9b_complete_source_paths": int(complete.sum()),
        "all_mapped_claims_have_complete_source_paths": bool(complete[mapped].all()),
        "all_wp9b_manifest_inputs_exist": True,
    }


# =============================================================================
# Frozen methodological stage specification
# =============================================================================


def methodology_stage_records() -> list[dict[str, object]]:
    """
    Return the deterministic stage-level methodological specification.

    These are framework-design classifications, not empirical estimates.
    """
    return [
        {
            "stage_id": "S01",
            "stage_name": "Corpus definition and acquisition",
            "stage_group": "input_design",
            "purpose": "Define the organisational units, textual population, inclusion rules, and acquisition procedure.",
            "primary_inputs": "domain corpus; organisational labels; document metadata",
            "primary_outputs": "auditable raw corpus and sampling frame",
            "domain_specificity": "high",
            "transferability": "low",
            "required_modification": "substantive_domain_redesign",
            "dependency_level": "domain_critical",
            "automation_level": "partial",
            "reproducibility_requirement": "explicit acquisition protocol and retained source metadata",
            "reusable_without_change": False,
            "configuration_items": "organisation definition|document type|inclusion window|language rules|licensing constraints",
            "rationale": "The analytical framework can be reused, but the population, labels, access conditions, and sampling logic must be redesigned for the target corpus.",
        },
        {
            "stage_id": "S02",
            "stage_name": "Corpus preparation and eligibility filtering",
            "stage_group": "data_preparation",
            "purpose": "Standardise text and metadata and apply documented analysis-eligibility rules.",
            "primary_inputs": "raw corpus and metadata",
            "primary_outputs": "analysis-ready corpus with eligibility flags",
            "domain_specificity": "medium",
            "transferability": "high",
            "required_modification": "minor_adaptation",
            "dependency_level": "schema_dependent",
            "automation_level": "high",
            "reproducibility_requirement": "versioned cleaning rules, eligibility criteria, and deterministic identifiers",
            "reusable_without_change": False,
            "configuration_items": "text fields|metadata schema|missingness rules|deduplication keys|eligibility criteria",
            "rationale": "Cleaning operations are generic, while field mappings and eligibility rules require transparent domain adaptation.",
        },
        {
            "stage_id": "S03",
            "stage_name": "Semantic representation",
            "stage_group": "representation",
            "purpose": "Represent each eligible document in a shared semantic feature space.",
            "primary_inputs": "analysis-ready text",
            "primary_outputs": "document-level semantic representations",
            "domain_specificity": "low",
            "transferability": "high",
            "required_modification": "configuration_only",
            "dependency_level": "model_and_language_dependent",
            "automation_level": "high",
            "reproducibility_requirement": "fixed model identifier, preprocessing version, inference settings, and random seeds where applicable",
            "reusable_without_change": False,
            "configuration_items": "representation model|language coverage|sequence length|batching|normalisation",
            "rationale": "The operation is domain-agnostic, but the selected representation must be appropriate for the language and document type.",
        },
        {
            "stage_id": "S04",
            "stage_name": "Semantic structure identification",
            "stage_group": "structure_discovery",
            "purpose": "Identify recurring semantic structure using a fixed similarity, graph, and community-analysis protocol.",
            "primary_inputs": "document-level semantic representations",
            "primary_outputs": "semantic communities and structural assignments",
            "domain_specificity": "low",
            "transferability": "high",
            "required_modification": "configuration_only",
            "dependency_level": "data_scale_dependent",
            "automation_level": "high",
            "reproducibility_requirement": "fixed construction parameters, algorithm versions, seeds, and sensitivity protocol",
            "reusable_without_change": False,
            "configuration_items": "similarity rule|neighbourhood parameter|community algorithm|resolution|minimum community size",
            "rationale": "The computational logic transfers directly; only scale-sensitive parameters require documented configuration.",
        },
        {
            "stage_id": "S05",
            "stage_name": "Organisational semantic profiling",
            "stage_group": "descriptive_analysis",
            "purpose": "Summarise how organisational units distribute across the identified semantic structure.",
            "primary_inputs": "community assignments; organisational labels",
            "primary_outputs": "company profiles, diversity measures, similarity contrasts, and community contributions",
            "domain_specificity": "low",
            "transferability": "high",
            "required_modification": "configuration_only",
            "dependency_level": "label_schema_dependent",
            "automation_level": "high",
            "reproducibility_requirement": "fixed metric definitions and explicit aggregation level",
            "reusable_without_change": False,
            "configuration_items": "organisational unit field|aggregation level|metric set|minimum sample thresholds",
            "rationale": "Profile construction is generic once organisational labels and semantic assignments are available.",
        },
        {
            "stage_id": "S06",
            "stage_name": "Statistical differentiation assessment",
            "stage_group": "validation",
            "purpose": "Assess whether observed organisational differences are supported beyond descriptive summaries.",
            "primary_inputs": "document- or artist-level profile metrics and organisational labels",
            "primary_outputs": "global, pairwise, enrichment, and uncertainty summaries",
            "domain_specificity": "low",
            "transferability": "high",
            "required_modification": "minor_adaptation",
            "dependency_level": "sampling_structure_dependent",
            "automation_level": "high",
            "reproducibility_requirement": "predeclared unit of analysis, test family, correction procedure, and resampling settings",
            "reusable_without_change": False,
            "configuration_items": "independent unit|test assumptions|multiple-testing correction|bootstrap unit|minimum group size",
            "rationale": "The validation logic is generic, but the independent unit and sampling hierarchy must match the target corpus.",
        },
        {
            "stage_id": "S07",
            "stage_name": "Temporal dynamics assessment",
            "stage_group": "longitudinal_analysis",
            "purpose": "Characterise semantic drift, stability, trajectories, and change points over time.",
            "primary_inputs": "dated semantic assignments and organisational labels",
            "primary_outputs": "drift, stability, trajectory, and change-point summaries",
            "domain_specificity": "medium",
            "transferability": "medium",
            "required_modification": "minor_adaptation",
            "dependency_level": "temporal_granularity_dependent",
            "automation_level": "high",
            "reproducibility_requirement": "fixed time unit, window definition, minimum coverage, and comparison rules",
            "reusable_without_change": False,
            "configuration_items": "time field|window width|minimum observations|consecutive comparison rule|coverage policy",
            "rationale": "The methods transfer, but meaningful temporal windows depend on corpus density and domain cadence.",
        },
        {
            "stage_id": "S08",
            "stage_name": "Robustness and sensitivity assessment",
            "stage_group": "validation",
            "purpose": "Test whether principal findings persist under plausible analytical perturbations.",
            "primary_inputs": "frozen outputs and predefined perturbation scenarios",
            "primary_outputs": "finding-level robustness, structural stability, and ranking-sensitivity records",
            "domain_specificity": "low",
            "transferability": "high",
            "required_modification": "minor_adaptation",
            "dependency_level": "pipeline_configuration_dependent",
            "automation_level": "high",
            "reproducibility_requirement": "predeclared perturbations, fixed comparison criteria, and complete availability reporting",
            "reusable_without_change": False,
            "configuration_items": "sampling perturbations|temporal windows|structure parameters|availability rules|conclusion vocabulary",
            "rationale": "The audit pattern transfers directly, while the relevant perturbations must correspond to the target pipeline's principal assumptions.",
        },
        {
            "stage_id": "S09",
            "stage_name": "Integrated organisational identity synthesis",
            "stage_group": "interpretation",
            "purpose": "Combine frozen descriptive, temporal, robustness, and identifiability evidence into an interpretable organisational fingerprint.",
            "primary_inputs": "validated outputs from preceding analytical stages",
            "primary_outputs": "integrated identity statements and evidence-linked summaries",
            "domain_specificity": "medium",
            "transferability": "medium",
            "required_modification": "substantive_domain_redesign",
            "dependency_level": "interpretive_ontology_dependent",
            "automation_level": "partial",
            "reproducibility_requirement": "explicit claim schema, controlled vocabulary, deterministic evidence mappings, and retained provenance",
            "reusable_without_change": False,
            "configuration_items": "identity dimensions|claim vocabulary|interpretive thresholds|domain terminology|evidence mapping rules",
            "rationale": "The synthesis architecture transfers, but the interpretive ontology and wording must be redesigned without changing the underlying evidence.",
        },
        {
            "stage_id": "S10",
            "stage_name": "Cross-work-package validation and traceability",
            "stage_group": "framework_validation",
            "purpose": "Audit agreement, provenance completeness, broken links, and evidence coverage across the full framework.",
            "primary_inputs": "frozen claims, evidence ledgers, source manifests, and work-package summaries",
            "primary_outputs": "consistency classifications, provenance network, coverage matrices, and integrity checks",
            "domain_specificity": "low",
            "transferability": "high",
            "required_modification": "configuration_only",
            "dependency_level": "output_schema_dependent",
            "automation_level": "high",
            "reproducibility_requirement": "stable identifiers, controlled evidence statuses, exact source paths, and deterministic integrity rules",
            "reusable_without_change": False,
            "configuration_items": "claim categories|work-package names|metric relevance map|evidence status vocabulary|source schema",
            "rationale": "The validation logic is generic and requires only schema mapping to a new implementation.",
        },
    ]


def build_methodology_stage_inventory() -> pd.DataFrame:
    columns = [
        "stage_id",
        "stage_name",
        "stage_group",
        "purpose",
        "primary_inputs",
        "primary_outputs",
        "domain_specificity",
        "dependency_level",
        "automation_level",
        "reproducibility_requirement",
    ]
    return pd.DataFrame(methodology_stage_records())[columns]


def build_stage_transferability_assessment() -> pd.DataFrame:
    records = pd.DataFrame(methodology_stage_records())
    columns = [
        "stage_id",
        "stage_name",
        "stage_group",
        "domain_specificity",
        "transferability",
        "required_modification",
        "dependency_level",
        "automation_level",
        "reusable_without_change",
        "configuration_items",
        "rationale",
    ]
    result = records[columns].copy()
    result["transferability_sort"] = result["transferability"].map(
        TRANSFERABILITY_ORDER
    )
    result["modification_sort"] = result["required_modification"].map(
        MODIFICATION_ORDER
    )
    result = result.sort_values(
        ["transferability_sort", "modification_sort", "stage_id"]
    ).drop(columns=["transferability_sort", "modification_sort"])
    return result.reset_index(drop=True)


# =============================================================================
# Derived audit tables
# =============================================================================


def build_framework_transferability_summary(
    assessment: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for dimension, ordered_values in [
        ("transferability", ["high", "medium", "low"]),
        (
            "required_modification",
            [
                "none",
                "configuration_only",
                "minor_adaptation",
                "substantive_domain_redesign",
            ],
        ),
        ("domain_specificity", ["low", "medium", "high"]),
        ("automation_level", ["high", "partial", "low"]),
    ]:
        counts = assessment[dimension].value_counts(dropna=False)
        for value in ordered_values:
            rows.append(
                {
                    "summary_dimension": dimension,
                    "summary_category": value,
                    "n_stages": int(counts.get(value, 0)),
                    "stage_ids": pipe_join(
                        assessment.loc[assessment[dimension] == value, "stage_id"]
                    ),
                    "stage_names": pipe_join(
                        assessment.loc[assessment[dimension] == value, "stage_name"]
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_domain_dependency_matrix(assessment: pd.DataFrame) -> pd.DataFrame:
    dependency_dimensions = [
        (
            "corpus_population_and_sampling",
            {"S01": "critical", "S02": "moderate"},
            "Population definition and acquisition are domain-specific; eligibility rules require schema adaptation.",
        ),
        (
            "language_and_document_form",
            {"S02": "moderate", "S03": "critical", "S09": "moderate"},
            "Representation choice and interpretive terminology must match language and document form.",
        ),
        (
            "organisational_label_schema",
            {"S01": "critical", "S05": "critical", "S06": "moderate", "S10": "moderate"},
            "The organisational unit must be defined consistently across profiling, validation, and provenance.",
        ),
        (
            "sampling_hierarchy",
            {"S02": "moderate", "S06": "critical", "S08": "moderate"},
            "The independent unit and resampling hierarchy depend on how documents are nested within creators or organisations.",
        ),
        (
            "temporal_cadence",
            {"S01": "moderate", "S07": "critical", "S08": "moderate"},
            "Window definitions and sensitivity analyses depend on corpus density and temporal cadence.",
        ),
        (
            "semantic_structure_scale",
            {"S03": "moderate", "S04": "critical", "S05": "moderate", "S08": "moderate"},
            "Structure-discovery configuration depends on corpus size, density, and semantic heterogeneity.",
        ),
        (
            "interpretive_ontology",
            {"S05": "moderate", "S09": "critical", "S10": "moderate"},
            "Identity dimensions and claim language require explicit domain redesign while preserving empirical provenance.",
        ),
        (
            "software_and_parameter_configuration",
            {"S03": "critical", "S04": "critical", "S06": "moderate", "S07": "moderate", "S08": "critical", "S10": "critical"},
            "Versions, parameters, seeds, and controlled vocabularies must be fixed for reproducible transfer.",
        ),
    ]

    stage_lookup = assessment.set_index("stage_id")["stage_name"].to_dict()
    rows: list[dict[str, object]] = []
    for dependency, stage_map, adaptation_note in dependency_dimensions:
        for stage_id in assessment["stage_id"]:
            rows.append(
                {
                    "dependency_dimension": dependency,
                    "stage_id": stage_id,
                    "stage_name": stage_lookup[stage_id],
                    "dependency_strength": stage_map.get(stage_id, "none"),
                    "requires_target_domain_decision": stage_map.get(stage_id, "none")
                    in {"critical", "moderate"},
                    "adaptation_note": adaptation_note,
                }
            )
    return pd.DataFrame(rows).sort_values(["dependency_dimension", "stage_id"])


def build_transferability_readiness(
    assessment: pd.DataFrame,
    validation: dict[str, object],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for row in assessment.itertuples(index=False):
        blockers: list[str] = []
        prerequisites: list[str] = []

        if row.stage_id == "S01":
            blockers.append("target corpus and organisational population not yet defined")
            prerequisites.extend(
                [
                    "document target-domain population",
                    "define organisational labels",
                    "record acquisition and licensing constraints",
                ]
            )
        elif row.stage_id == "S09":
            blockers.append("target-domain identity ontology not yet defined")
            prerequisites.extend(
                [
                    "define target-domain identity dimensions",
                    "create controlled claim vocabulary",
                    "predeclare evidence-to-claim mapping rules",
                ]
            )
        else:
            prerequisites.extend(
                [
                    item.replace("_", " ")
                    for item in normalise_text(row.configuration_items).split("|")
                    if item
                ]
            )

        if row.required_modification == "configuration_only":
            readiness = "ready_after_configuration"
        elif row.required_modification == "minor_adaptation":
            readiness = "ready_after_documented_adaptation"
        elif row.required_modification == "substantive_domain_redesign":
            readiness = "requires_domain_design"
        else:
            readiness = "ready_without_change"

        rows.append(
            {
                "stage_id": row.stage_id,
                "stage_name": row.stage_name,
                "transferability": row.transferability,
                "required_modification": row.required_modification,
                "readiness_status": readiness,
                "methodological_logic_reusable": True,
                "current_kpop_implementation_frozen": True,
                "upstream_validation_available": bool(
                    validation["all_mapped_claims_have_complete_source_paths"]
                ),
                "target_domain_prerequisites": pipe_join(prerequisites),
                "known_blockers_before_transfer": pipe_join(blockers),
                "requires_new_empirical_validation_after_transfer": True,
            }
        )

    return pd.DataFrame(rows)


def build_reusable_pipeline_components(assessment: pd.DataFrame) -> pd.DataFrame:
    component_records = [
        ("C01", "deterministic document identifiers", "S02", "reuse_pattern", "direct_reuse", "Preserve stable document and organisational identifiers throughout the pipeline."),
        ("C02", "analysis-eligibility flag", "S02", "data_contract", "schema_mapping", "Retain an explicit Boolean eligibility field rather than deleting excluded records."),
        ("C03", "semantic representation interface", "S03", "computational_interface", "configuration", "Accept analysis-ready text and emit one semantic representation per eligible document."),
        ("C04", "semantic structure construction protocol", "S04", "algorithmic_protocol", "configuration", "Retain the graph/community workflow with fixed, documented scale-sensitive parameters."),
        ("C05", "organisational profile tables", "S05", "output_schema", "schema_mapping", "Retain company-by-community profiles, diversity summaries, and pairwise contrasts."),
        ("C06", "independent-unit statistical validation", "S06", "validation_protocol", "minor_adaptation", "Retain global and pairwise validation while redefining the appropriate independent unit."),
        ("C07", "rolling temporal comparison framework", "S07", "longitudinal_protocol", "minor_adaptation", "Retain drift and stability logic while redefining the time unit and coverage thresholds."),
        ("C08", "finding-level robustness ledger", "S08", "audit_schema", "minor_adaptation", "Retain one row per tested finding, perturbation, conclusion, and source file."),
        ("C09", "evidence-linked identity claim schema", "S09", "interpretive_schema", "domain_redesign", "Retain stable claim IDs and evidence mappings while redesigning the target-domain vocabulary."),
        ("C10", "cross-work-package evidence ledger", "S10", "provenance_schema", "schema_mapping", "Retain supporting, qualifying, conflicting, and unavailable evidence statuses."),
        ("C11", "traceability node-edge representation", "S10", "provenance_structure", "direct_reuse", "Retain deterministic Company-to-Claim-to-Evidence-to-Source paths."),
        ("C12", "input and run manifests", "S10", "reproducibility_artifact", "direct_reuse", "Retain exact paths, availability checks, row counts, and integrity outcomes for each run."),
    ]

    stage_lookup = assessment.set_index("stage_id")[
        ["stage_name", "transferability", "required_modification"]
    ].to_dict("index")

    rows: list[dict[str, object]] = []
    for component_id, name, stage_id, component_type, reuse_mode, note in component_records:
        stage = stage_lookup[stage_id]
        rows.append(
            {
                "component_id": component_id,
                "component_name": name,
                "stage_id": stage_id,
                "stage_name": stage["stage_name"],
                "component_type": component_type,
                "reuse_mode": reuse_mode,
                "stage_transferability": stage["transferability"],
                "stage_required_modification": stage["required_modification"],
                "reuse_note": note,
            }
        )
    return pd.DataFrame(rows)


def build_framework_reuse_checklist() -> pd.DataFrame:
    checks = [
        ("R01", "scope", "Target organisational population is explicitly defined.", "before_data_acquisition", True, "S01"),
        ("R02", "scope", "Document types, time period, language, and inclusion criteria are fixed.", "before_data_acquisition", True, "S01|S02"),
        ("R03", "governance", "Data access, licensing, privacy, and retention constraints are documented.", "before_data_acquisition", True, "S01"),
        ("R04", "data_contract", "Stable document, creator, organisation, and time identifiers are available.", "before_preparation", True, "S02"),
        ("R05", "data_contract", "Analysis eligibility is recorded explicitly and deterministically.", "before_preparation", True, "S02"),
        ("R06", "representation", "The semantic representation model is suitable for the target language and document form.", "before_representation", True, "S03"),
        ("R07", "representation", "Model version, preprocessing, inference settings, and seeds are frozen.", "before_representation", True, "S03"),
        ("R08", "structure", "Similarity and community-analysis parameters are predeclared.", "before_structure_discovery", True, "S04"),
        ("R09", "profiling", "The organisational unit and aggregation level are fixed.", "before_profiling", True, "S05"),
        ("R10", "statistics", "The independent observational unit and sampling hierarchy are justified.", "before_inference", True, "S06"),
        ("R11", "statistics", "Global, pairwise, enrichment, correction, and resampling rules are predeclared.", "before_inference", True, "S06"),
        ("R12", "temporal", "Time units, windows, minimum coverage, and comparison rules are justified.", "before_temporal_analysis", False, "S07"),
        ("R13", "robustness", "Perturbations correspond to the principal assumptions of the transferred pipeline.", "before_robustness_analysis", True, "S08"),
        ("R14", "interpretation", "Target-domain identity dimensions and claim vocabulary are defined before synthesis.", "before_identity_synthesis", True, "S09"),
        ("R15", "interpretation", "Every identity claim has deterministic evidence-mapping rules.", "before_identity_synthesis", True, "S09|S10"),
        ("R16", "traceability", "Each evidence record retains source work package, file, and source-row provenance.", "throughout_pipeline", True, "S10"),
        ("R17", "traceability", "Broken edges, orphan nodes, unavailable analyses, and unmapped claims are reported.", "framework_validation", True, "S10"),
        ("R18", "reproducibility", "Exact software versions, parameters, random seeds, inputs, and outputs are archived.", "throughout_pipeline", True, "S03|S04|S06|S07|S08|S10"),
        ("R19", "validation", "Transferred empirical findings are revalidated rather than inherited from the K-pop application.", "after_transfer", True, "S05|S06|S07|S08|S09|S10"),
        ("R20", "reporting", "Transfer claims distinguish reusable logic from target-domain empirical validity.", "publication", True, "S01|S09|S10"),
    ]

    return pd.DataFrame(
        checks,
        columns=[
            "check_id",
            "check_category",
            "check_description",
            "required_stage",
            "mandatory_for_transfer",
            "related_stage_ids",
        ],
    )


def build_run_summary(
    assessment: pd.DataFrame,
    reusable: pd.DataFrame,
    checklist: pd.DataFrame,
    validation: dict[str, object],
) -> pd.DataFrame:
    values: dict[str, object] = {
        "n_methodological_stages": int(len(assessment)),
        "n_high_transferability_stages": int((assessment["transferability"] == "high").sum()),
        "n_medium_transferability_stages": int((assessment["transferability"] == "medium").sum()),
        "n_low_transferability_stages": int((assessment["transferability"] == "low").sum()),
        "n_configuration_only_stages": int((assessment["required_modification"] == "configuration_only").sum()),
        "n_minor_adaptation_stages": int((assessment["required_modification"] == "minor_adaptation").sum()),
        "n_domain_redesign_stages": int((assessment["required_modification"] == "substantive_domain_redesign").sum()),
        "n_reusable_pipeline_components": int(len(reusable)),
        "n_reuse_checklist_items": int(len(checklist)),
        "n_mandatory_checklist_items": int(checklist["mandatory_for_transfer"].sum()),
        "n_wp9a_mapped_claims_verified": int(validation["n_wp9a_mapped_claims"]),
        "n_wp9b_claims_verified": int(validation["n_wp9b_claims"]),
        "all_mapped_claims_have_complete_source_paths": bool(validation["all_mapped_claims_have_complete_source_paths"]),
        "all_upstream_manifest_inputs_exist": bool(validation["all_wp9b_manifest_inputs_exist"]),
        "new_embeddings_created": False,
        "new_graph_created": False,
        "new_clustering_created": False,
        "new_prediction_model_created": False,
        "new_latent_variable_created": False,
        "new_inferential_statistics_created": False,
        "assessment_type": "deterministic_methodological_audit",
        "overall_transferability_conclusion": "analytical_core_highly_transferable_with_domain_specific_input_and_interpretive_redesign",
    }

    return pd.DataFrame(
        [{"metric": key, "value": value} for key, value in values.items()]
    )


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = resolve_inputs()
    input_manifest = build_input_manifest(paths)
    data = load_inputs(paths)
    validation = validate_frozen_inputs(data)

    inventory = build_methodology_stage_inventory()
    assessment = build_stage_transferability_assessment()
    framework_summary = build_framework_transferability_summary(assessment)
    dependency_matrix = build_domain_dependency_matrix(assessment)
    readiness = build_transferability_readiness(assessment, validation)
    reusable = build_reusable_pipeline_components(assessment)
    checklist = build_framework_reuse_checklist()
    run_summary = build_run_summary(
        assessment=assessment,
        reusable=reusable,
        checklist=checklist,
        validation=validation,
    )

    outputs = {
        "methodology_stage_inventory.csv": inventory,
        "stage_transferability_assessment.csv": assessment,
        "framework_transferability_summary.csv": framework_summary,
        "domain_dependency_matrix.csv": dependency_matrix,
        "transferability_readiness.csv": readiness,
        "reusable_pipeline_components.csv": reusable,
        "framework_reuse_checklist.csv": checklist,
        "transferability_input_manifest.csv": input_manifest,
        "transferability_run_summary.csv": run_summary,
    }

    for filename, dataframe in outputs.items():
        dataframe.to_csv(OUTPUT_DIR / filename, index=False)

    n_high = int((assessment["transferability"] == "high").sum())
    n_medium = int((assessment["transferability"] == "medium").sum())
    n_low = int((assessment["transferability"] == "low").sum())
    n_config = int(
        (assessment["required_modification"] == "configuration_only").sum()
    )
    n_minor = int(
        (assessment["required_modification"] == "minor_adaptation").sum()
    )
    n_redesign = int(
        (
            assessment["required_modification"]
            == "substantive_domain_redesign"
        ).sum()
    )

    print("=" * 88)
    print("WP9c METHODOLOGICAL TRANSFERABILITY COMPLETE")
    print("=" * 88)
    print(f"Project root: {ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Methodological stages assessed: {len(assessment)}")
    print(f"High transferability stages: {n_high}")
    print(f"Medium transferability stages: {n_medium}")
    print(f"Low transferability stages: {n_low}")
    print(f"Configuration-only stages: {n_config}")
    print(f"Minor-adaptation stages: {n_minor}")
    print(f"Domain-redesign stages: {n_redesign}")
    print(f"Reusable pipeline components: {len(reusable)}")
    print(f"Reuse checklist items: {len(checklist)}")
    print(
        "Mapped claims with complete upstream source paths: "
        f"{validation['all_mapped_claims_have_complete_source_paths']}"
    )


if __name__ == "__main__":
    main()
